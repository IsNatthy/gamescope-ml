from pathlib import Path

from cleaning.header_fixer import HeaderFixer


PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = PROJECT_ROOT / "data" / "raw" / "games.csv"
FIXED_FILE = PROJECT_ROOT / "data" / "raw" / "games_fixed.csv"

def main():
    fixer = HeaderFixer(
        input_file=INPUT_FILE,
        output_file=FIXED_FILE
    )

    fixer.fix_header()
    fixer.show_summary()


if __name__ == "__main__":
    main()