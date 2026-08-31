from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

from neosyntropy.tools.coding.coding_tools import CodingTools


def main() -> None:
    with TemporaryDirectory(prefix="neosyntropy-filesystem-") as temp_dir:
        base_dir = Path(temp_dir)
        registry = CodingTools(
            base_dir=str(base_dir),
            enable_read_file=True,
            enable_write_file=True,
            enable_find=True,
            enable_ls=True,
            enable_edit_file=False,
            enable_run_shell=False,
        ).register()

        write_notes = registry.invoke(
            "write_file",
            {
                "file_path": "notes.txt",
                "contents": "NeoSyntropy cookbook\n- write files\n- read files\n- keep it local\n",
            },
        ).result
        write_summary = registry.invoke(
            "write_file",
            {
                "file_path": "summary.md",
                "contents": "# Summary\n\nThis example uses the coding filesystem toolkit.\n",
            },
        ).result

        print(write_notes)
        print(write_summary)
        print()

        print("List the temporary workspace:")
        print(registry.invoke("ls", {"path": None, "limit": 20}).result)
        print()

        print("Read back `notes.txt`:")
        print(registry.invoke("read_file", {"file_path": "notes.txt", "offset": 0, "limit": 20}).result)
        print()

        print("Read back `summary.md`:")
        print(registry.invoke("read_file", {"file_path": "summary.md", "offset": 0, "limit": 20}).result)


if __name__ == "__main__":
    main()
