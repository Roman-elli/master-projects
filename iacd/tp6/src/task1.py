import sys
import shlex
sys.modules['pipes'] = shlex 

from mrjob.job import MRJob
from mrjob.step import MRStep

class MRFriendsByAge(MRJob):

    def steps(self):
        return [
            MRStep(mapper=self.mapper_get_ages,
                   reducer=self.reducer_avg_friends)
        ]

    def mapper_get_ages(self, _, line):
        (user_id, name, age, num_friends) = line.split(',')
        yield int(age), float(num_friends)

    def reducer_avg_friends(self, age, num_friends_list):
        total_friends = 0.0
        num_elements = 0
        for friends in num_friends_list:
            total_friends += friends
            num_elements += 1
        yield age, round(total_friends / num_elements, 2)

if __name__ == '__main__':
    MRFriendsByAge.run()