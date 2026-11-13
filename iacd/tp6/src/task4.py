import sys
import shlex
sys.modules['pipes'] = shlex

from mrjob.job import MRJob
from mrjob.step import MRStep

class MRSpendByCustomerSorted(MRJob):

    def steps(self):
        return [
            MRStep(mapper=self.mapper_get_orders,
                   reducer=self.reducer_totals_by_customer),
            MRStep(mapper=self.mapper_make_amounts_key,
                   reducer=self.reducer_output_results)
        ]

    def mapper_get_orders(self, _, line):
        # Campos: Customer_ID, Product_ID, Amount
        (customer_id, product_id, order_amount) = line.split(',')
        yield int(customer_id), float(order_amount)

    def reducer_totals_by_customer(self, customer_id, orders):
        yield customer_id, sum(orders)

    def mapper_make_amounts_key(self, customer_id, total_amount):
        # Formata o valor decimal com zeros à esquerda (ex: 03309.38) para ordenação crescente
        yield '%04.02f' % float(total_amount), customer_id

    def reducer_output_results(self, total_amount, customer_ids):
        for customer_id in customer_ids:
            yield customer_id, round(float(total_amount), 2)

if __name__ == '__main__':
    MRSpendByCustomerSorted.run()