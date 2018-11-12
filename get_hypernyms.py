import warnings
warnings.filterwarnings("ignore", message="numpy.dtype size changed, may indicate binary incompatibility. Expected 96, got 88")
from nltk.corpus import wordnet as wn
from pprint import pprint


locations = ['indoor.a.01', 'outside.r.01', 'outdoor.a.01', 'inside.r.01',
             'water_system.n.02', 'sidewalk.n.01', 'field.n.01',
             'stadium.n.01', 'city.n.01', 'booth.n.02', 'garage.n.01',
             'hill.n.01', 'road.n.01', 'room.n.01', 'facade.n.01',
             'building.n.01', 'curb.n.01', 'entrance.n.01', 'outside.n.01',
             'shop.n.01', 'land.n.04', 'outdoors.n.01', 'crossing.n.05', 'carnival.n.03',
             'skyscraper.n.01', 'lane.n.01', 'cell.n.06', 'floor.n.01', 'theater.n.01',
             'office.n.01', 'balcony.n.01', 'barroom.n.01', 'parking.n.01', 'museum.n.01',
             'puddle.n.01', 'inside.n.01', 'house.n.01', 'outside.n.01', 'pool.n.01',
             'beach.n.01', 'ocean.n.01', 'forest.n.01', 'lake.n.01', 'park.n.01',
             'hallway.n.01', 'tunnel.n.01', 'restaurant.n.01', 'cafe.n.01', 'yard.n.01',
             'metro.n.01', 'airport.n.01', 'garden.n.01', 'factory.n.01', 'lawn.n.01',
             'supermarket.n.01', 'university.n.01', 'highway.n.01', 'school.n.01',
             'hospital.n.01', 'court.n.01', 'sea.n.01', 'mountain.n.01', 'cliff.n.01']

temporal = ['day.n.01', 'summer.n.01', 'spring.n.01', 'evening.n.01',
            'afternoon.n.01']


def get_hypernyms(synset):
    """Returns top level category for the synset given"""

    term_list = []

    for s in synset.hypernyms():
        term_list += [s.name()]
        h = get_hypernyms(s)
        if len(h):
            term_list += [h]

    return term_list + [synset.name()]


def print_hypernyms(synsets):
    for syn in synsets:
        hypernyms = get_hypernyms(wn.synset(syn))

        print('Synset: {}'.format(syn))
        pprint(hypernyms)
        print('-' * 80)


def main():
    print("Printing hypernyms for:")
    print("=" * 80)
    print("Locations")
    print("=" * 80)
    print_hypernyms(locations)
    print("=" * 80)
    print("Temporal")
    print("=" * 80)
    print_hypernyms(temporal)


if __name__ == "__main__":
    main()
