import replicate
from dotenv import load_dotenv
load_dotenv()

def main():
    output = replicate.run(
        "jy2575/tri_s1_s2_13b:8c50a12a176c2e887a4fcf28495d75d733d7cc72c3a6f5cf68a36d6e96d9ff6a",
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

    # The jy2575/tri_s1_s2_13b model can stream output as it's running.
    # The predict method returns an iterator, and you can iterate over that output.
    for item in output:
        # https://replicate.com/jy2575/tri_s1_s2_13b/api#output-schema
        print(item, end="")


if __name__ == "__main__":
    main()

