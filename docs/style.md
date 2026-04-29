# Style Guide for `xeed.py`

This document outlines the coding style and architectural conventions observed
in the `xeed.py` script. The goal is to ensure that future modifications are
consistent with the existing codebase.

## 0. Goals

*   **Single file:** `xeed.py` must implement its own logic and tests.
*   **Hermetic:** `xeed.py` cannot rely on packages outside the Python stdlib.
*   **Backwards compatible:** `xeed.py` must support Python 3.10.0.
*   **Object-oriented:** The codebase aggressively leverages Python's
    object-oriented features, using classes to encapsulate related data and
    behavior, and promoting modular design.

## 1. Naming Conventions

*   **Variables:** Use `snake_case` for local variables (e.g., `final_command`).
*   **Functions:** Use `snake_case` for function names (e.g.,
    `build_command_from_configs`).
*   **Classes:** Use `PascalCase` for class names (e.g., `Blob`, `ToolChest`).
*   **Constants & Globals:** Use `UPPER_SNAKE_CASE` for module-level constants
    and globals (e.g., `LOG`, `DEFAULT_CONFIG`, `ENV`).
*   **Methods:** Use `snake_case` for class methods (e.g., `get_path`).
*   **"Private" Methods:** Use a single leading underscore for internal helper
    methods that are not part of the public class API (e.g., `_merge_configs`).

## 2. Code Structure & Architecture

*   **Modularity:** The script is organized into a set of cohesive classes, each
    responsible for a specific concern (e.g., `Cli` for argument parsing, `Blob`
    for data structure management, `CfgConfig` for configuration parsing).
*   **Composition over Inheritance:** The architecture favors composition and
    multiple inheritance for mixing functionality. For example, `XeedCache` is
    composed of `HashedCache` and `FileCache`.
*   **Factory Pattern:** Use `@classmethod` to provide factory methods for
    object creation, especially when creating instances from external sources
    like files or dictionaries (e.g., `from_path`, `from_blob`).
*   **Main Function:** The `main()` function acts as an orchestrator. It should
    not contain complex logic directly but should instead instantiate and call
    methods on the various classes to drive the application.
*   **Entry Point:** The script must be executable and use the standard `if
    __name__ == "__main__":` guard to call the `main()` function.

## 3. Formatting and Whitespace

*   **Indentation:** Use 4 spaces for indentation.
*   **Line Length:** Keep lines to a reasonable length (under 100 characters is
    a good guideline) to maintain readability.
*   **Blank Lines:**
    *   Use two blank lines to separate top-level functions and class
        definitions.
    *   Use one blank line to separate methods within a class.
*   **Whitespace:**
    *   Use a single space around operators (`=`, `==`, `+`, etc.).
    *   Avoid extraneous whitespace.

## 4. Comments and Docstrings

*   **Docstrings:** Docstrings are not currently used. The code is expected to
    be largely self-documenting through clear naming of classes, methods, and
    variables.
*   **Comments:** Use inline comments (`#`) sparingly. They should explain the
    *why* behind a complex or non-obvious piece of code, not the *what*.

## 5. Python Features

*   **Decorators:** Decorators are used extensively and are the preferred way to
    add cross-cutting functionality.
    *   `@classmethod`: For factories.
    *   `@functools.cached_property`: For memoizing property lookups that are
        expensive to compute.
    *   `@functools.wraps`: In custom decorators to preserve the original
        function's metadata.
*   **Type Hinting:** Type hints are not currently used.
*   **f-strings:** f-strings are preferred for string formatting where values
    are directly available. For more complex template-like formatting,
    `string.Formatter` is used.
*   **Testing:** Tests are written using the `pytest` framework and are included
    at the end of the same file as the application code. Test functions should
    be named `test_*`.
*   **Imports:** Imports should be grouped at the top of the file in the
    following order:
    1.  Standard library imports.
    2.  Third-party library imports.
    3.  Local application/module imports.
