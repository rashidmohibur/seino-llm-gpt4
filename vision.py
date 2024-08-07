import base64
import json
import os

from langchain.prompts import PromptTemplate

VISION_TEMPlATE = """
Your task is to extract information from the below invoice document. The text is obtained from OCR of a scanned document.
The quality of the OCR is not perfect, and there may be errors in the text. Fix any errors you find.
If you are not sure about the correct value, you can leave it empty.
You MUST response in JSON follows the format below. No need to use ```json.
{
    "invoice_amount": "", // 金額: the total amount of the invoice
    "date": "", // 日付: the issue date of the invoice. Format: YYYY-MM-DD.
    "recipient": "", // 請求先会社名: the name of the recipient of the invoice.
    "issuer_name": "", // 請求元会社名: the name of the issuer of the invoice. Be careful not to confuse with the recipient company and the shipper. Issuer company usually appears on the right side of the document.
    "ship_name": "", // 船名, Flight No: the name of the ship or flight that carries the goods
    "shipper": "" // 荷主: refers to the person or company that sends goods for shipment
}

Example response:
{
    "invoice_amount": "1000",
    "date": "2022-01-01",
    "recipient": "Recipient Co., Ltd.",
    "issuer_name": "Issuer Co., Ltd.",
    "ship_name": "Ship Name",
    "shipper": "Shipper Co., Ltd."
}

## Result:
"""


class VisionBasedExtractor:
    def __init__(self, llm):
        self.llm = llm

    def extract(self, img_path):

        base64_image = self._encode_image(img_path)

        messages = [
            {
                "role": "system",
                "content": "You are an AI assistant that helps people extract information from document image.",
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": VISION_TEMPlATE,
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                    },
                ],
            },
        ]
        result = self.llm.invoke(messages)

        try:
            json_data = json.loads(result.content)
            return json_data
        except json.JSONDecodeError:
            print("Failed to parse LLM result")
            print(result.content)
            return None

    def _encode_image(self, image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")
