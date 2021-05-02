import praw
import os
os.environ['HTTP_PROXY'] = os.environ['HTTPS_PROXY'] = 'http://localhost:12333'

reddit = praw.Reddit("bot1", config_interpolation="basic")

hot_posts = reddit.subreddit('wallstreetbets').hot(limit=10)
for post in hot_posts:
    print(post.title, end='\n\n')
