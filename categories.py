# Top level categories:

Categories = {
    'location': [
        # Synsets that are hypernyms of frequently found synsets:
        'way.n.06', 'excavation.n.03', 'geological_formation.n.01', 'show.n.01',
        'facility.n.01', 'establishment.n.04',
        'geological_formation.n.01', 'body_of_water.n.01', 'vegetation.n.01',
        'railway.n.01', 'body.n.02', 'region.n.03', 'area.n.05', 'building.n.01',
        'building_complex.n.01', 'institution.n.01',
        # Synsets that occur as they are:
        'indoor.a.01', 'outside.r.01', 'outdoor.a.01', 'inside.r.01',
        'stadium.n.01', 'facade.n.01', 'curb.n.01', 'land.n.04', 'floor.n.01', 'parking.n.01',
        'balcony.n.01', 'university.n.01', 'outdoors.n.01'],
    'temporal': ['time_unit.n.01', 'time_period.n.01']
}


if __name__ == "__main__":
    from nltk.corpus import wordnet as wn
    print("=" * 80)
    print("Current categories:")
    print("=" * 80)
    for cat in Categories:
        print("Top level synsets for {}".format(cat))
        print("=" * 80)
        for syn in Categories[cat]:
            print('{} - {}'.format(syn, wn.synset(syn).definition()))
        print("=" * 80)
