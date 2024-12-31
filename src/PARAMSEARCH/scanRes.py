with open('param_search.bash', 'r') as file:
    debias_lines = [line.strip() for line in file if 'run_debias_mlm.py' in line]

with open('evaluations.txt', 'r') as file:
    results_lines = file.readlines()
# for line in debias_lines:
#     lr = line.split('--learning_rate')[1].split()[0]
#     alpha, beta = map(lambda x: round(float(x), 1), line.split('--weighted_loss')[1].split()[:2])
#     debias_layer = line.split('--debias_layer')[1].split()[0]
#     loss_target = line.split('--loss_target')[1].split()[0]
#     print(f"Learning Rate: {lr}, Weighted Loss: {alpha} {beta}, Debias Layer: {debias_layer}, Loss Target: {loss_target}")
#

# for i, line in enumerate(debias_lines):
#     seat_res = results_lines[i].split('seat_res: ')[1].split()[0]
#     heilman_res_str = results_lines[i].split('heilman_res: ')[1].strip(')\n').replace('(', '').replace(')', '')
#     heilman_res = tuple(map(float, heilman_res_str.split(', ')))
#
#     lr = line.split('--learning_rate')[1].split()[0]
#     alpha, beta = map(lambda x: round(float(x), 1), line.split('--weighted_loss')[1].split()[:2])
#     debias_layer = line.split('--debias_layer')[1].split()[0]
#     loss_target = line.split('--loss_target')[1].split()[0]
#
#     print(
#         f"Learning Rate: {lr}, Weighted Loss: {alpha} {beta}, Debias Layer: {debias_layer}, Loss Target: {loss_target}, Seat Res: {seat_res}, Heilman Res: {heilman_res}")

optimal_results = {}
wlSixFour = {}
wlThreeSeven = {}
for line in debias_lines:
    lr = float(line.split('--learning_rate')[1].split()[0])
    alpha, beta = map(float, line.split('--weighted_loss')[1].split()[:2])
    debias_layer = line.split('--debias_layer')[1].split()[0]
    loss_target = line.split('--loss_target')[1].split()[0]

    seat_res = float(results_lines[debias_lines.index(line)].split('seat_res: ')[1].split()[0].strip('[[]],'))
    heilman_res = tuple(map(float, results_lines[debias_lines.index(line)].split('heilman_res: ')[1].strip(')\n').replace('(', '').replace(')',
                                                                                                                    '').split(
        ', ')))


    combined_res = heilman_res + (seat_res, )

    if alpha == 0.6 and beta == 0.4 and lr == 5e-05:
        wlSixFour[(alpha, beta, debias_layer, loss_target)] = ((lr, alpha, beta), sum(combined_res), combined_res)

    if alpha == 0.3 and beta == 0.7 and lr == 5e-05:
        wlThreeSeven[(alpha, beta, debias_layer, loss_target)] = ((lr, alpha, beta), sum(combined_res), combined_res)

    # Check if combination exists in the dictionary and update if necessary
    if (debias_layer, loss_target) not in optimal_results or (
            sum(combined_res) < optimal_results[(debias_layer, loss_target)][1]):
        optimal_results[(debias_layer, loss_target)] = ((lr, alpha, beta), sum(combined_res), combined_res)

# Print the optimal results
for (debias_layer, loss_target), ((lr, alpha, beta), combined_res, res) in optimal_results.items():
    print(
        f"Debias Layer: {debias_layer}, Loss Target: {loss_target}, Learning Rate: {lr}, Weighted Loss: {alpha} {beta}, Heilman Res: {res[:3]}, seat res: {res[3]}")

 # == pick one of these
print(wlSixFour)
print(wlThreeSeven)