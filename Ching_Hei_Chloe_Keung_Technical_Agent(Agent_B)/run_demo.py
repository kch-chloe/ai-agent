# run_demo.py
"""
Thin runner script.
Calls main() defined in Technical_Agent_Chloe.py.
"""

from Technical_Agent_Chloe import main, build_azure_client

if __name__ == "__main__":
    client = build_azure_client()
    main(client=client)
