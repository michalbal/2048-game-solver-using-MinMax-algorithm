34242923
31632765
*****
Comments:
    According to the hint we went with linear combination of features = "qualities" * "weights"
    Where the qualities are:
    To maximize: number of empty tiles and the fact that the biggest tile is in the right lower
    corner.
    To minimize: Break of monotonicity (Right + Down).
    The idea is that while trying to keep numbers in monotonically rising order in rows to the right
    and in columns downwards, we are trying to always have the biggest tile in the same corner,
    surrounded with heavy concentration of high-value tiles.
    Secondary condition is the empty space on the board as game_over is precisely when the board is
    full.
    For better readability and comparability (to the human eye) the weights lead to the same order
    of magnitude ("1.3f" where used for the fine-tuning).

    A bit further on the weights in the function description.