from datetime import datetime, timedelta
from urllib.request import urlopen as u
from os import mkdir
from os.path import exists
from time import sleep

from glob import glob
import pandas as pd
from json import loads

'''
Download the raw data in JSON format
'''
def download():
	coins = {
		'Bitcoin': 1,
		'Ethereum': 1027,
		'Cardano': 2010,
		'Binance Coin': 1839,
		'XRP': 52,
		'Solana': 5426,
		'Polkadot': 6636,
		'Dogecoin': 74,
		'Avalanche': 5805,
		'Terra': 4172,
		'Litecoin': 2,
		'Bitcoin Cash': 1831,
		'Algorand': 4030,
		'Filecoin': 2280,
		'Cosmos': 3794,
		'Internet Computer': 8916,
		'Polygon': 3890,
		'TRON': 1958,
		'Stellar': 512,
		'Ethereum Classic': 1321
		'VeChain': 3077
	}


	for coin in coins:
		if not exists(f'coindata/{coin}'):
			mkdir(f'coindata/{coin}')
		
		starts = datetime(2018,12,31,0,0)

		while starts.year != 2021 or starts.month != 9 or starts.day != 22:
			starts += timedelta(days = 1)
			ends = starts + timedelta(days = 1)

			startsEpoch = int(starts.timestamp())
			endsEpoch = int(ends.timestamp())

			saveFileName = starts.strftime('%Y%m%d')

			if exists(f'coindata/{coin}/{saveFileName}.json'):
				print(f'Skipped coindata/{coin}/{saveFileName}.json')
				continue

			v = None
			while v is None:
				try:
					v = u(f'https://api.coinmarketcap.com/data-api/v3/cryptocurrency/detail/chart?id={coins[coin]}&range={startsEpoch}~{endsEpoch}')
				except:
					pass
			
			with open(f'coindata/{coin}/{saveFileName}.json','w') as f:
				f.write(v.read().decode(encoding='utf-8'))
				print(f'Written to coindata/{coin}/{saveFileName}.json')


'''
Transform the raw data into readable CSV/JSON
'''
def extract():
	# df = pd.DataFrame(columns=('Currency', 'Date/Time', 'Price USD', 'Trading Volume Last 24h', 'Market Cap', 'Price BTC', 'DunnoWhatThisIs'))
	data = {}

	i = 0

	for g in glob('coindata\\*\\*.json'):
		
		currency = g.split('\\')[1]
		with open(g, 'r') as f:
			try:
				mydata = loads(f.read())
				for x in mydata['data']['points']:
					data[i] = {'Currency':currency, 'DateTime':pd.to_datetime(datetime.fromtimestamp(int(x)).strftime('%Y-%m-%d %H:%M:%S')), 'Price USD':mydata['data']['points'][x]['v'][0],
					'Trading Volume Last 24h':mydata['data']['points'][x]['v'][1], 'Market Cap':mydata['data']['points'][x]['v'][2], 'Price BTC':mydata['data']['points'][x]['v'][3], 'DunnoWhatThisIs':mydata['data']['points'][x]['v'][4]}
					i += 1
			except:
				pass
		print(f'Finished processing {g}')

	df = pd.DataFrame.from_dict(data, orient='index')

	print('Done.')

	df.sort_values(['DateTime'],inplace=True)
	df.to_json('historical_coin_data_5m.json')
	df.to_csv('historical_coin_data_5m.csv')

if __name__ == '__main__':
	download()
	extract()
