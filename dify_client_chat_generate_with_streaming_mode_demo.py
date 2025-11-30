import json
import os
import time
import uuid

from dify_client import ChatClient

dify_api_key = None


def init_dify_api_key():
    global dify_api_key

    if "DIFY_API_KEY" in os.environ:
        dify_api_key = os.environ["DIFY_API_KEY"]
    else:
        dify_api_key = "YOUR_API_KEY"


if __name__ == "__main__":
    init_dify_api_key()

    # Initialize ChatClient
    chat_client = ChatClient(
        api_key=dify_api_key,
        base_url="http://100.85.209.38/v1",
    )

    # Queries
    queries = [
        "你好。",
        "我想报修一下网络。",
        "学生公寓六号楼401房间无线网络无法连接。",
    ]

    user_id = str(uuid.uuid4())
    conversation_id = None
    for query in queries:
        # Create Chat Message using ChatClient
        chat_response = chat_client.create_chat_message(
            inputs={"timestamp": int(time.time()), "phone": "13800138000"},
            query=query,
            user=user_id,
            response_mode="streaming",
            conversation_id=conversation_id,
        )
        chat_response.raise_for_status()

        for line in chat_response.iter_lines():
            line = line.split("data:", 1)[-1]
            if not line.strip():
                continue

            line = json.loads(line.strip())
            if line.get("event") != "message":
                continue

            if conversation_id is None:
                conversation_id = line["conversation_id"]

            print(line)

        time.sleep(1)
