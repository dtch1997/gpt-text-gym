def remove_leading_whitespace(string: str, max_remove: int = -1) -> str:
    lines = string.split("\n")
    new_lines = []
    for line in lines:
        # Remove leading whitespace
        if max_remove == -1:
            line = line.lstrip()
        else:
            remove = min(max_remove, len(line) - len(line.lstrip()))
            line = line[remove:]
        new_lines.append(line)
    return "\n".join(new_lines)
