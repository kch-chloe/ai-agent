"""
run_demo.py
Force-runs KO strategy script from top to bottom.
"""

from pathlib import Path

SCRIPT_PATH = Path(__file__).parent / "Ma_Suet_Nam_Technical_Agent(Agent_A)" / "Ma_Suet_Nam_Technical_Agent.py"

def main():
    print("=== Running KO strategy demo ===")
    code = SCRIPT_PATH.read_text()
    exec(compile(code, SCRIPT_PATH.name, "exec"), {})
    print("=== Demo completed successfully ===")

if __name__ == "__main__":
    main()
