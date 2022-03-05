import argparse
import pandas as pd
from pmaw import PushshiftAPI
import datetime as dt

import os


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
            self.sublist.extend([s for s in self.submissions])
        self.subdf = pd.DataFrame(self.sublist)
        self.subdf.to_csv(path)
        print(f'Saved {len(self.sublist)} pieces of Submissions to "{path}".')

    def getcom(self, args, fields, path):
        keywords = [
            ['GME', 'GameStop'],
            ['amc'],
            ['Tesla', 'TSLA'],
            ['Apple', "AAPL"],
            ['Koss'],
            ['BlackBerry', 'BB'],
            ['Tilray', 'TLRY'],
            ['Palantir', 'PLTR'],
            ['Nokia', 'NOK']
        ]  # 关键词合集，大小写不敏感

        # added multi-keyword support
        times = [
            dt.datetime(2021, 1, 1, 0, 0),
            dt.datetime(2021, 2, 1, 0, 0),
            dt.datetime(2021, 3, 1, 0, 0),
            dt.datetime(2021, 4, 1, 0, 0),
            dt.datetime(2021, 5, 1, 0, 0),
            dt.datetime(2021, 6, 1, 0, 0),
            dt.datetime(2021, 7, 1, 0, 0),
            dt.datetime(2021, 8, 1, 0, 0),
            dt.datetime(2021, 9, 1, 0, 0),
            dt.datetime(2021, 10, 1, 0, 0),
            dt.datetime(2021, 11, 1, 0, 0),
            dt.datetime(2021, 12, 1, 0, 0),
            dt.datetime(2022, 1, 1, 0, 0),
        ]
        times = [int(i.timestamp()) for i in times]
        for keyword in keywords:
            self.comlist = []
            for q in keyword:
                for i in range(len(times)-1):
                    self.comments = self.api.search_comments(
                        after=times[i],
                        before=times[i+1],
                        fields=fields,
                        limit=int(args.comments_limit/len(keyword)),
                        mem_safe=True,
                        q=q,
                        safe_exit=True,
                        sort_type=args.comments_sort_type,
                        subreddit=args.subreddit
                    )
                    self.comlist.extend([c for c in self.comments])
                    print(self.comlist[-1])
            self.comdf = pd.DataFrame(self.comlist)
            self.comdf.to_csv(f'{path}/{keyword[0]}.csv')
            print(
                f'Saved {len(self.comlist)} pieces of Comments to "{path}/{keyword[0]}.csv".')

    def len_com(self):
        return len(self.comlist)

    def len_sub(self):
        return len(self.sublist)

    def save(self, subfile='sub.csv', comfile='com.csv'):
        self.subdf.to_csv(subfile)
        self.comdf.to_csv(comfile)
