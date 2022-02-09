import argparse
import pandas as pd
from pmaw import PushshiftAPI

import os
os.environ['HTTP_PROXY'] = os.environ['HTTPS_PROXY'] = 'http://localhost:3898'


class Model:
    def __init__(self):
        self.api = PushshiftAPI()

    def getsub(self, args, fields, path):
        # Get submissions under r/wallstreetbets, includes GME
        self.sublist = []
        for q in args.q:
            self.submissions = self.api.search_submissions(
                after=args.after,
                before=args.before,
                fields=fields,
                limit=args.submissions_limit,
                mem_safe=True,
                num_comments=args.num_comments,
                q=args.q,
                safe_exit=True,
                sort_type=args.submissions_sort_type,
                subreddit=args.subreddit
            )
            print('\n', end='')
            self.sublist.extend([s for s in self.submissions])
        self.subdf = pd.DataFrame(self.sublist)
        self.subdf.to_csv(path)
        print(f'Saved {len(self.sublist)} pieces of Submissions to "{path}".')

    def getcom(self, args, fields, path):
        # Get comments under r/wallstreetbets, includes GME
        # added multi-keyword support
        self.comlist = []
        for q in args.q:
            self.comments = self.api.search_comments(
                after=args.after,
                before=args.before,
                fields=fields,
                limit=args.comments_limit,
                mem_safe=True,
                q=q,
                safe_exit=True,
                sort_type=args.comments_sort_type,
                subreddit=args.subreddit
            )
            print('\n', end='')
            self.comlist.extend([c for c in self.comments])
        self.comdf = pd.DataFrame(self.comlist)
        self.comdf.to_csv(path)
        print(f'Saved {len(self.comlist)} pieces of Comments to "{path}".')

    def len_com(self):
        return len(self.comlist)

    def len_sub(self):
        return len(self.sublist)

    def save(self, subfile='sub.csv', comfile='com.csv'):
        self.subdf.to_csv(subfile)
        self.comdf.to_csv(comfile)
