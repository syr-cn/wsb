# By SYR-XJTU
# Imports
import datetime as dt
import argparse
from Model import *


# PARAMETERS
parser = argparse.ArgumentParser()
# time range
parser.add_argument('-before', type=int,
                    default=int(dt.datetime(2021, 4, 1, 0, 0).timestamp()))
parser.add_argument('-after', type=int,
                    default=int(dt.datetime(2020, 12, 1, 0, 0).timestamp()))
# subreddit
parser.add_argument('-q', type=str, default='GME')
parser.add_argument('-subreddit', type=str, default='wallstreetbets')
parser.add_argument('-submissions_sort_type', type=str, default='num_comments',
                    choices=['created_utc', 'score', 'num_comments'])
parser.add_argument('-comments_sort_type', type=str, default='score',
                    choices=['created_utc', 'score', 'num_comments'])
# limits
parser.add_argument('-submissions_limit', type=int, default=200)
parser.add_argument('-comments_limit', type=int, default=500)
parser.add_argument('-num_comments', type=str, default='>100')
# response filter
fields = (
    'created_utc',
    'full_link',
    'id',
    'num_comments',
    'num_crossposts',
    'score',
    'selftext',
    'title',
    'upvote_ratio'
)

args = parser.parse_args()

if __name__ == '__main__':
    model = Model(args, fields)
    print(model.len_com())
    print(model.len_sub())
    model.save()