import sys
import getopt
import dataset
import hnn



def main(argv):
    image_path = ''
    classifier = 'cnn'

    try:
        opts, args = getopt.getopt(argv, 'hi:c:', ['help=', 'image=', 'classifier='])
    except getopt.GetoptError:
        print('\nusage: python predict.py -i <image>\n')
        sys.exit(2)

    for opt, arg in opts:
        if opt in ('-h', '--help'):
            print('\nusage: python predict.py -i <image>')
            print('\nCommands:')
            print('\t-h [--help],       : Show help')
            print('\t-i [--image],      : Path to the image')
            print('\t-c [--classifier], : Specify classifier. CNN (default) or ANN')
            sys.exit()
        elif opt in ('-i', '--image'):
            image_path = arg
        else:
            classifier = arg.lower()

    print('\n=> Initializing classifier ...')


    print('=> Predicting ...')

    dset = dataset.HiraSet('dset', 22500)
    hiraNet = hnn.HNN(classifier, classifier)
    model = hiraNet.load()
    t = model.predict(image_path)
    rez_t = t.argmax(axis=1)
    print('\nPrediction:' % dset.entries()[rez_t].label())

if __name__ == '__main__':
    main(sys.argv[1:])