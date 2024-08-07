import json
import os
import re

from langchain.prompts import PromptTemplate

FULLTEXT_TEMPlATE = """
Your task is to extract information from the below invoice document. The text is obtained from OCR of a scanned document.
The quality of the OCR is not perfect, and there may be errors in the text. Fix any errors you find.
If you are not sure about the correct value, you can leave it empty.
You MUST response in JSON follows the format below. No need to use ```json.

"invoice_amount": 0, // 金額: the total amount of the invoice. Format: number.
"date": "", // 日付: the issue date of the invoice. Format: string, YYYY-MM-DD.
"recipient": "", // 請求先会社名: the name of the recipient of the invoice. Format: string.
"issuer_name": "", // 請求元会社名: the name of the issuer of the invoice. Be careful not to confuse with the recipient company and the shipper. Issuer company usually appears on the right side of the document.
"ship_name": "", // 船名, Flight No: the name of the ship or flight that carries the goods
"shipper": "" // 荷主: refers to the person or company that sends goods for shipment

Example response:
{
    "invoice_amount": 1000,
    "date": "2022-01-01",
    "recipient": "Recipient Co., Ltd.",
    "issuer_name": "Issuer Co., Ltd.",
    "ship_name": "Ship Name",
    "shipper": "Shipper Co., Ltd."
}

## Text:
{{fulltext}}

## Result (In JSON format):
"""


class TextBasedExtractor:
    def __init__(self, text_detector, llm):
        self.text_detector = text_detector
        self.llm = llm

    def extract(self, img_path):
        fulltext = self.text_detector.extract_and_format_fulltext(
            "https://aigw-pre.fastaccounting.jp/document/v1/fulltext",
            os.getenv("AIGW_API_KEY"),
            img_path,
        )

        print(fulltext)

        prompt = PromptTemplate.from_template(
            FULLTEXT_TEMPlATE, template_format="jinja2"
        )

        query = prompt.format(fulltext=fulltext)

        result = self.llm.predict(query)

        # For claude
        result = result[result.find("{") : result.rfind("}") + 1]

        print(result)

        try:
            json_data = json.loads(result)
            return json_data
        except json.JSONDecodeError:
            print("Failed to parse LLM result")
            print(result)
            return None
