import argparse
import matplotlib.pyplot as plt


def load_data(filename, header_lines=0):
    """Reads a file with one number per line skipping header lines."""

    with open(filename, 'rt') as file:
        while header_lines > 0:
            next(file)
        data = [float(line) for line in file]

    return data


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plotting Utility')
    parser.add_argument('--logscale', action='store_true', help='Plot y-axis on log scale')
    parser.add_argument('-s', '--save', type=str, help='Filename for saving plot')
    parser.add_argument('-t', '--title', type=str, help='Title for the plot')
    parser.add_argument('file', nargs='+', help='Data files for plotting')

    args = parser.parse_args()
    plot_fcn = plt.semilogy if args.logscale else plt.plot

    plt.figure(1)
    min_value = None
    for filename in args.file:
        print("Reading data from {}...".format(filename))
        data = load_data('loss.txt')
        plot_fcn(data)
        if min_value is None:
            min_value = min(data)
        else:
            min_value = min([min_value, min(data)])

    plot_fcn([0, len(data)], [min(data), min(data)], 'r--')
    plt.grid()

    # title
    if args.title is not None:
        plt.title(args.title)
    elif len(args.file) == 1:
        plt.title(args.file[0])

    # save or show
    if args.save is not None:
        print("Saving plot to {}...".format(args.save))
        plt.savefig(args.save)
    else:
        plt.show()
