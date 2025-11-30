import os

import dashscope


def init_dashscope_api_key():
    if "DASHSCOPE_API_KEY" in os.environ:
        dashscope.api_key = os.environ["DASHSCOPE_API_KEY"]
    else:
        dashscope.api_key = "YOUR_API_KEY"


dify_api_key = None


def init_dify_api_key():
    global dify_api_key

    if "DIFY_API_KEY" in os.environ:
        dify_api_key = os.environ["DIFY_API_KEY"]
    else:
        dify_api_key = "YOUR_API_KEY"


def main():
    init_dashscope_api_key()
    init_dify_api_key()


if __name__ == "__main__":
    main()
