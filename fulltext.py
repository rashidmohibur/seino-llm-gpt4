from typing import List

import requests
from pydantic import BaseModel


class Rectangle(BaseModel):
    x1: int
    y1: int
    x2: int
    y2: int


class Record(BaseModel):
    value: str
    position: Rectangle
    xmid: float
    ymid: float
    char_width: int
    confidence: float


class Row(BaseModel):
    value: str
    position: Rectangle
    records: List[Record]


class TextDetector:
    def __init__(self, url, api_key):
        self.url = url
        self.api_key = api_key

    def extract_fulltext(self, url, api_key, file_path) -> List[Record]:
        header = {"X-Api-Key": api_key}
        response = requests.post(
            url, headers=header, files={"file": open(file_path, "rb")}
        )

        data = response.json()
        records = data["data"]

        result = []
        for rec in records:
            if rec["value"] == "":
                continue
            result.append(
                Record(
                    value=rec["value"],
                    position=Rectangle(**rec["position"]),
                    xmid=(rec["position"]["x1"] + rec["position"]["x2"]) / 2,
                    ymid=(rec["position"]["y1"] + rec["position"]["y2"]) / 2,
                    char_width=int(
                        (rec["position"]["x2"] - rec["position"]["x1"])
                        / len(rec["value"])
                    ),
                    confidence=rec["confidence"],
                )
            )

        formatted = self._grouping(result)

        return formatted

    def extract_and_format_fulltext(self, url, api_key, file_path) -> str:
        records = self.extract_fulltext(url, api_key, file_path)
        return self._format_layout(records)

    def _format_layout(self, rows: List[Row]) -> str:
        min_char_width = 999999
        for row in rows:
            for item in row.records:
                min_char_width = min(min_char_width, item.char_width)

        result = ""
        for row in rows:
            row_text = ""
            prev_pos = 0
            prev_item_len = 0
            prev_char_width = min_char_width
            for item in row.records:
                space_added = (
                    int(
                        (
                            item.position.x1
                            - prev_pos
                            # - (prev_char_width - min_char_width) * prev_item_len
                        )
                        / min_char_width
                    )
                    // 2
                )
                if space_added < 0:
                    space_added = 0
                row_text += " " * space_added + item.value

                prev_pos = item.position.x2
                prev_char_width = item.char_width
                prev_item_len = len(item.value)
            result += f"{row_text}\n"

        return result

    # grouping combines text records into rows.
    def _grouping(self, records: List[Record]) -> List[Row]:
        rows = []
        selected = [False] * len(records)
        MaxRatio = 0.1  # Define MaxRatio value

        for i, record in enumerate(records):
            if selected[i]:
                continue

            rr = [record]
            selected[i] = True

            # Search forward
            j = i
            while j < len(records):
                if selected[j]:
                    j += 1
                    continue
                if self._same_row(record, records[j], MaxRatio):
                    rr.append(records[j])
                    selected[j] = True
                j += 1

            # Search backward
            j = i
            while j >= 0:
                if selected[j]:
                    j -= 1
                    continue
                if self._same_row(record, records[j], MaxRatio):
                    rr.append(records[j])
                    selected[j] = True
                j -= 1

            # Sort position left to right
            rr.sort(key=lambda x: x.xmid)

            row_text = ""
            rects = []
            for item in rr:
                row_text += item.value
                rects.append(item.position)

            revised = self._revise_row_text(row_text)
            if revised != "":
                rows.append(Row(value=revised, records=rr, position=rects[0]))

        return rows

    def _revise_row_text(self, value: str) -> str:
        return value.replace(" ", "")

    def _same_row(self, record1: Record, record2: Record, max_ratio: float) -> bool:
        return (
            abs(
                (record2.ymid - record1.ymid)
                / float(record1.position.y2 - record1.position.y1)
            )
            < max_ratio
        )
