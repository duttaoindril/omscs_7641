import numpy as np

days = 60

if __name__ == "__main__":
    stock_data = np.genfromtxt('../stocks/ge.us.txt', delimiter=',')[1:, -3]
    print(stock_data)
    stock_data_rearranged = []
    stock_data_rearranged_normalized = []
    for i in range(len(stock_data) - (days + 1)):
        a = np.around(stock_data[i : i + days], 4)
        e = (a - np.mean(a)) / np.std(a)
        # print("ROW: ", a)
        today = stock_data[i + days - 1]
        tomorrow = stock_data[i + days]
        # print("TODAY: ", today)
        # print("TOMORROW: ", tomorrow)
        diff = tomorrow - today
        # print("DIFF: ", diff)
        buy_sell_hold = 'BUY' if diff > 0 else ('SELL' if diff < 0 else 'HOLD')
        # print("LABEL: ", buy_sell_hold)
        # print("=================")
        stock_data_rearranged.append(np.append(a, buy_sell_hold))
        stock_data_rearranged_normalized.append(np.append(e, buy_sell_hold))
    print(stock_data_rearranged)
    np.savetxt("stock.csv", stock_data_rearranged, delimiter=",", fmt='%s')
    np.savetxt("stock_normalized.csv", stock_data_rearranged_normalized, delimiter=",", fmt='%s')
