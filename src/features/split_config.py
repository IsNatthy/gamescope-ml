"""Configuracion y utilidades compartidas para la particion train/test."""

from sklearn.model_selection import train_test_split

TEST_SIZE = 0.2
RANDOM_STATE = 42


def split_indices(row_count: int) -> tuple[list[int], list[int]]:
    """Devuelve indices reproducibles de train y test."""
    train, test = train_test_split(
        range(row_count), test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    return list(train), list(test)
