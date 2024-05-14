import replicate
from dotenv import load_dotenv
load_dotenv()

def main():
    output = replicate.run(
        "jy2575/tri_s1_s4:b89e95ca503efb72c2ef8800e7797bb53cf8533ec7d56861fd9825e04a3f27e3",
        input={
            "debug": False,
            "top_k": 50,
            "top_p": 0.9,
            "temperature": 0.75,
            "system_prompt": "You are a helpful assistant.",
            "max_new_tokens": 128,
            "min_new_tokens": -1,
            "prompt": "who plays ross?"
        }
    )

    # The jy2575/tri_s1_s4 model can stream output as it's running.
    # The predict method returns an iterator, and you can iterate over that output.
    for item in output:
        # https://replicate.com/jy2575/tri_s1_s4/api#output-schema
        print(item, end="")


if __name__ == "__main__":
    main()

