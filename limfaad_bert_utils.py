"""
Shared utilities for BERT on LIMFAAD (tabular → text).
Converts a preprocessed LIMFAAD row into a fixed text format for BERT.
Used by both training and inference so the representation is identical.
"""

import pandas as pd

FEATURE_COLUMNS = [
    'Followers', 'Following', 'Following/Followers', 'Posts', 'Posts/Followers',
    'Bio', 'Profile Picture', 'External Link', 'Mutual Friends', 'Threads'
]


def row_to_text(row: pd.Series) -> str:
    """
    Convert a single LIMFAAD row (after preprocessing) to a text sentence for BERT.
    Handles both numeric (0/1) and string (yes/no) encodings for binary fields.
    """
    def _bool_str(val):
        if isinstance(val, str):
            return val.strip().lower() in ('yes', 'y', '1')
        return int(val) != 0

    followers = int(row.get('Followers', 0))
    following = int(row.get('Following', 0))
    ff_ratio = float(row.get('Following/Followers', 0))
    posts = int(row.get('Posts', 0))
    pf_ratio = float(row.get('Posts/Followers', 0))
    bio = 'yes' if _bool_str(row.get('Bio', 0)) else 'no'
    profile_pic = 'yes' if _bool_str(row.get('Profile Picture', 0)) else 'no'
    external = 'yes' if _bool_str(row.get('External Link', 0)) else 'no'
    mutual = int(row.get('Mutual Friends', 0))
    threads = 'yes' if _bool_str(row.get('Threads', 0)) else 'no'

    return (
        f"Account has {followers} followers and {following} following. "
        f"Following to followers ratio is {ff_ratio:.4f}. "
        f"Posts count is {posts}. Posts per follower ratio is {pf_ratio:.4f}. "
        f"Bio: {bio}. Profile picture: {profile_pic}. External link: {external}. "
        f"Mutual friends: {mutual}. Threads: {threads}."
    )
