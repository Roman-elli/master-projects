from pathlib import Path

# Path variables
DATA_PATH = Path().cwd() / ".." / "assets" / "raw_data.csv"
SAVE_DATA_PATH = Path().cwd() / ".." / "assets" / "clean_data.csv"

# Data extraction / cleaning variables
unused_columns = [
        'Payment Method', 'Cancelled Rides by Customer', 
        'Cancelled Rides by Driver', 'Incomplete Rides'
    ]

important_columns = ['Date', 'Time', 'Booking ID', 'Booking Status', 'Pickup Location']

cancel_fill_map = {
        'Reason for cancelling by Customer': 'Não Cancelado pelo Cliente',
        'Driver Cancellation Reason': 'Não Cancelado pelo Condutor',
        'Incomplete Rides Reason': 'Viagem Completa (Sem problemas)'
    }

numeric_columns = [
        'Booking Value', 'Ride Distance', 'Driver Ratings', 
        'Customer Rating', 'Avg VTAT', 'Avg CTAT'
    ]