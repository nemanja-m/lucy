COLORS = dict(
    red='31m',
    green='32m',
    yellow='33m',
    blue='34m',
    white='39m'
)


def colorize(text, color='green', bold=True):
    default_color = COLORS.get('green')
    color_code = COLORS.get(color, default_color)
    bold_char = str(int(bold)) + ';'
    return '\033[{}{}{}\033[0m'.format(bold_char, color_code, text)
