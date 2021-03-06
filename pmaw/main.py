# By SYR-XJTU
# Imports
import datetime as dt
import argparse
from Model import *


# PARAMETERS
parser = argparse.ArgumentParser()
# time range
parser.add_argument('-before', type=int,
                    default=int(dt.datetime(2021, 12, 31, 0, 0).timestamp()))
parser.add_argument('-after', type=int,
                    default=int(dt.datetime(2021, 1, 1, 0, 0).timestamp()))
# subreddit

# parser.add_argument('-q', type=list, default=['GME', 'Gamestop'])
# q 为关键字列表，大小写不敏感

parser.add_argument('-subreddit', type=str, default='wallstreetbets')
parser.add_argument('-submissions_sort_type', type=str, default='num_comments',
                    choices=['created_utc', 'score', 'num_comments'])
parser.add_argument('-comments_sort_type', type=str, default='score',
                    choices=['created_utc', 'score', 'num_comments'])
# limits
parser.add_argument('-submissions_limit', type=int, default=3000)
parser.add_argument('-comments_limit', type=int, default=1200)
# 获取的评论条数下限
parser.add_argument('-num_comments', type=str, default='>10')
# response filter
subfields = (
    'author',
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
comfields = (
    'author',
    'body',
    'created_utc',
    'score',
    'total_awards_received'
)

args = parser.parse_args()

if __name__ == '__main__':
    # os.environ['HTTP_PROXY'] = os.environ['HTTPS_PROXY'] = 'http://localhost:3898'
    os.environ['http_proxy'] = os.environ['https_proxy'] = 'http://localhost:3898'
    model = Model()
    # model.getsub(args, subfields,'sub.csv')
    model.getcom(args, comfields, '~/Music/wsb/pmaw/3_1-12000')
