# Development Tools

This directory contains utility scripts and tools used during development of the HelixZone project. These tools are **not part of the distributed software** but are maintained in the repository for development purposes.

## Included Tools

- **check_task.py**: Debugging utility to check the status of a specific task by ID
- **list_tasks.py**: Lists all tasks in the system and creates a test task for debugging
- **search_all_files.py**: Searches all project files for a specific identifier
- **search_for_task.py**: Specialized tool to search for task identifiers in logs and source files

## Usage

These tools are designed to be run from the project root directory. Example:

```bash
python dev_tools/list_tasks.py
```

## Adding New Tools

When adding new development tools:

1. Place them in this directory
2. Update this README with a brief description
3. Make sure the tools import from the main codebase correctly (using relative imports)

## Note

These tools are excluded from distribution by the `.gitignore` file, but the directory structure is preserved to maintain organization. 