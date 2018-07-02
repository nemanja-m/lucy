COLORS = dict(
    red='31m',
    green='32m',
    yellow='33m',
    blue='34m',
    white='39m'
)


def colorize(text, color='green', bold=True):
    """Colors terminal text.

    Escapes input text with ANSI escape sequences to provide colored text output
    on terminals.

    Args:
        text (str): Text for coloring.
        color (str, optional): Color name. Supported colors are red, green, blue,
            yellow and white. Default: green.
        bold (bool, optional): Use bold text or not. Default: True.

    Returns:
      ANSI escaped colored string.
    """
    default_color = COLORS.get('green')
    color_code = COLORS.get(color, default_color)
    bold_char = str(int(bold)) + ';'
    return '\033[{bold}{color}{text}\033[0m'.format(bold=bold_char,
                                                    color=color_code,
                                                    text=text)
