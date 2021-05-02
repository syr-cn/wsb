import argparse
import pandas as pd
from pmaw import PushshiftAPI

import os
os.environ['HTTP_PROXY'] = os.environ['HTTPS_PROXY'] = 'http://localhost:12333'


class Model:
    def __init__(self):
        self.api = PushshiftAPI()

    def getsub(self, args, fields):
        # Get submissions under r/wallstreetbets, includes GME
        self.submissions = self.api.search_submissions(
            after=args.after,
            before=args.before,
            fields=fields,
            limit=args.submissions_limit,
            mem_safe=False,
            num_comments=args.num_comments,
            q=args.q,
            safe_exit=True,
            sort_type=args.submissions_sort_type,
            subreddit=args.subreddit
        )
        print('\n', end='')
        self.sublist = [s for s in self.submissions]
        self.subdf = pd.DataFrame(self.sublist)
        self.subdf.to_csv('sub.csv')
        print(f'Saved {len(self.comlist)} pieces of Submissions to "sub.csv".')

    def getcom(self, args, fields):
        # Get comments under r/wallstreetbets, includes GME
        self.comments = self.api.search_comments(
            after=args.after,
            before=args.before,
            fields=fields,
            limit=args.comments_limit,
            mem_safe=False,
            q=args.q,
            safe_exit=True,
            sort_type=args.comments_sort_type,
            subreddit=args.subreddit
        )
        print('\n', end='')
        self.comlist = [c for c in self.comments]
        self.comdf = pd.DataFrame(self.comlist)
        self.comdf.to_csv('com.csv')
        print(f'Saved {len(self.comlist)} pieces of Comments to "com.csv".')

    def len_com(self):
        return len(self.comlist)

    def len_sub(self):
        return len(self.sublist)

    def save(self, subfile='sub.csv', comfile='com.csv'):
        self.subdf.to_csv(subfile)
        self.comdf.to_csv(comfile)
