"""
Constants and current configuration of the 1.5 meter monitor
"""
# pylint: disable=invalid-name
from dataclasses import dataclass


@dataclass
class Color: #pylint: disable=too-many-instance-attributes
    """ Common colors """
    RED = (0, 0, 255)
    GREEN = (0, 255, 0)
    YELLOW = (0, 255, 255)
    BLUE = (255, 0, 0)
    WHITE = (255, 255, 255)
    GREY = (70, 70, 70)

    MINION_GREEN = (157, 187, 0)
    MINION_YELLOW = (51, 212, 244)
    MINION_RED = (57, 67, 208)


BANNER = "./media/banners/banner_1.jpg"
BANNER_WIDTH = 0.4

CLOSENESS_LEVELS = {
    0: {'name': 'safe', 'text': {'NL': 'Veilig', 'EN': 'Safe'},
        'color': Color.GREEN, 'dist': -1,
        'icon': "./media/icons/emoji/head_green.png"},
    1: {'name': 'low_risk', 'text': {'NL': 'Laag Risico', 'EN': 'Low Risk'},
        'color': Color.YELLOW, 'dist': 180,
        'icon': "./media/icons/emoji/head_yellow.png"},
    2: {'name': 'high_risk', 'text': {'NL': 'Hoog Risico', 'EN': 'High Risk'},
        'color': Color.RED, 'dist': 150,
        'icon': "./media/icons/emoji/head_red.png"},
}

CLOSENESS_LEVELS_MINION = {
    0: {'name': 'safe', 'text': {'NL': 'Veilig', 'EN': 'Safe'},
        'color': Color.MINION_YELLOW, 'dist': -1,
        'icon': "./media/icons/minions/minion_yellow.png"},
    1: {'name': 'low_risk', 'text': {'NL': 'Laag Risico', 'EN': 'Low Risk'},
        'color': Color.MINION_GREEN, 'dist': 180,
        'icon': "./media/icons/minions/minion_green.png"},
    2: {'name': 'high_risk', 'text': {'NL': 'Hoog Risico', 'EN': 'High Risk'},
        'color': Color.MINION_RED, 'dist': 150,
        'icon': "./media/icons/minions/minion_red.png"},
}

CLOSENESS_LEVELS_SIMPLE = {
    0: {'name': 'safe', 'text': {'NL': 'Veilig', 'EN': 'Safe'},
        'color': Color.GREEN, 'dist': -1,
        'icon': "./media/icons/simple/adam.png"},
    1: {'name': 'low_risk', 'text': {'NL': 'Laag Risico', 'EN': 'Low Risk'},
        'color': Color.YELLOW, 'dist': 180,
        'icon': "./media/icons/simple/adam.png"},
    2: {'name': 'high_risk', 'text': {'NL': 'Hoog Risico', 'EN': 'High Risk'},
        'color': Color.RED, 'dist': 150,
        'icon': "./media/icons/simple/adam.png"},
}

CALIBRATION_DISTANCE = 150

OUTPUT_FOLDER = 'output'
