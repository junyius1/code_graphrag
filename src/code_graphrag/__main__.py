"""Allow ``python -m code_graphrag ...`` (delegates to the Typer CLI)."""

from code_graphrag.cli.app import main

if __name__ == "__main__":
    main()
