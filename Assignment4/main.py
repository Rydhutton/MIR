import vamp as vp
import vampyhost as vh
import numpy as np
import librosa

def main():

    print vh.list_plugins()

    #segmentino:segmentino
    #qm-vamp-plugins:qm-segmenter

    #vh.load_plugin('qm-vamp-plugins:qm-segmenter', 44100, 0)
    x = vp.process_audio('Julian.mp3', 44100, 'qm-vamp-plugins:qm-segmenter', 'Julian.txt')

    #collect function
    #librosa and vamp collect
    vp.c



if __name__ == "__main__":
    main()