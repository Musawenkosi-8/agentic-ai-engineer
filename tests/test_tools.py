from src.tools import save_research_note


if __name__ == "__main__":

    result = save_research_note.invoke(
        {
            "filename": "../../../secret.txt",
            "content": "Bitcoin is currently trading around $65,000 according to recent market data."
        }
    )

    print("\n===== FILE SAVE RESULT =====")
    print(result)