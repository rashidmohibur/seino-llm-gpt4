import glob
import os
import time
from langchain_openai import AzureChatOpenAI
from vision import VisionBasedExtractor
from text import TextBasedExtractor
from fulltext import TextDetector
from langchain_aws import BedrockLLM

text_detector = TextDetector(
    "https://aigw-pre.fastaccounting.jp/document/v1/fulltext", os.getenv("AIGW_API_KEY")
)

image_dir = "/Users/luong.duy/TestData/30_seino/test_images"
image_paths = glob.glob(os.path.join(image_dir, "*.jpg"))

# llm = AzureChatOpenAI(
#     azure_endpoint="https://llm-poc-01.openai.azure.com/",
#     azure_deployment="sbcs01",
#     openai_api_version="2023-05-15",
#     openai_api_type="azure",
#     openai_api_key="b0ac364d5e5d45cb824e600d662a1cf4",
#     model_name="gpt-35-turbo",
#     temperature=0,
#     max_tokens=1000,
# )

llm = AzureChatOpenAI(
    azure_endpoint="https://for-fine-tuning.openai.azure.com/",
    azure_deployment="gpt4o",
    openai_api_version="2023-05-15",
    openai_api_type="azure",
    openai_api_key="fb8324ed98a943a2b62c6db6947e8e3b",
    model_name="gpt-4o",
    temperature=0,
    max_tokens=1000,
)

# llm = BedrockLLM(
#     credentials_profile_name="fa_prod", model_id="anthropic.claude-instant-v1",
#     region_name="ap-northeast-1"
# )

extractor = TextBasedExtractor(text_detector, llm)
# extractor = VisionBasedExtractor(llm)

result = ""
for img_path in image_paths:
    # if not "202402.pdf_01.jpg" in img_path:
    #     continue

    start_time = time.time()
    print(f"Processing {img_path}")
    data = extractor.extract(img_path)
    print(data)

    end_time = time.time()
    interval = end_time - start_time

    invoice_amount = data.get("invoice_amount", "")
    date = data.get("date", "")
    issuer_name = data.get("issuer_name", "")
    ship_name = data.get("ship_name", "")
    shipper = data.get("shipper", "")

    result += f"{img_path}\t{invoice_amount}\t{date}\t{issuer_name}\t{ship_name}\t{shipper}\t{interval}\n"

    break

with open("result/test.tsv", "w") as f:
    f.write(result)
