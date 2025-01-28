inputs = [
    {
        "optimal": True,
        "turns_to_go": 10,
        "map": [
            ['P', 'P', 'I', 'P'],
            ['P', 'P', 'I', 'P'],
            ['P', 'P', 'P', 'P'],
            ['P', 'P', 'I', 'P']
        ],
        "wizards": {'Harry Potter': {"location": (2, 0)}
                    },
        "horcrux": {'Nagini': {"location": (0, 3),
                               "possible_locations": ((0, 3), (1, 3), (2, 2)),
                               "prob_change_location": 0.9}
                    },
        "death_eaters": {'Lucius Malfoy': {"index": 0,
                                           "path": [(1, 1), (2, 1), (2, 2)]}},
    },

    {
        "optimal": True,
        "turns_to_go": 100,
        "map": [
            ['I', 'I', 'P'],
            ['P', 'I', 'P'],
            ['I', 'P', 'P'],
            ['P', 'I', 'P']
        ],
        "wizards": {'Harry Potter': {"location": (2, 2)}
                    },
        "horcrux": {'Nagini': {"location": (0, 2),
                               "possible_locations": ((0, 2), (1, 2), (2, 2)),
                               "prob_change_location": 0.9},
                    'Diary': {"location": (0, 0),
                              "possible_locations": ((0, 0), (1, 0), (2, 0)),
                              "prob_change_location": 0.3}
                    },
        "death_eaters": {'Snape': {"index": 1,
                                           "path": [(2, 2), (2, 1)]}},
    },
    {
        "optimal": True,
        "turns_to_go": 100,
        "map": [
            ['I', 'I', 'P'],
            ['P', 'I', 'P'],
            ['I', 'P', 'P'],
            ['P', 'I', 'P']
        ],
        "wizards": {'Harry Potter': {"location": (0, 2)}
                    },
        "horcrux": {'Nagini': {"location": (0, 2),
                                "possible_locations": ((0, 2), (1, 2), (2, 2)),
                                "prob_change_location": 0.9},
                    'Diary': {"location": (0, 0),
                                "possible_locations": ((0, 0), (2, 0)),
                                "prob_change_location": 0.45}
                    },
        "death_eaters": {'Snape': {"index": 1,
                                            "path": [(1, 0), (2, 1)]},
                        'random_de': {"index": 0,
                                    "path": [(3, 2)]}},
    },

    {
        "optimal": False,
        "turns_to_go": 30,
        "map": [
            ['P', 'P', 'I', 'P'],
            ['P', 'P', 'I', 'P'],
            ['P', 'P', 'P', 'P'],
            ['P', 'P', 'I', 'P']
        ],
        "wizards": {'Harry Potter': {"location": (2, 0)},
                    'Ron Weasley': {"location": (2, 1)}
                    },
        "horcrux": {'Nagini': {"location": (0, 3),
                               "possible_locations": ((0, 3), (1, 3), (2, 2)),
                               "prob_change_location": 0.4}
                    },
        "death_eaters": {'Lucius Malfoy': {"index": 0,
                                           "path": [(1, 1), (1, 0)]}},
    },

    {
        "optimal": False,
        "turns_to_go": 100,
        "map": [
            ['I', 'P', 'P', 'P', 'P', 'P', 'I'],
            ['P', 'P', 'P', 'P', 'P', 'P', 'P'],
            ['P', 'P', 'I', 'I', 'I', 'P', 'P'],
            ['P', 'P', 'I', 'P', 'I', 'P', 'P'],
            ['P', 'P', 'I', 'I', 'I', 'P', 'P'],
            ['P', 'P', 'P', 'P', 'P', 'P', 'P'],
            ['P', 'P', 'P', 'P', 'P', 'P', 'I']
        ],
        "wizards": {'Harry Potter': {"location": (2, 0)}
                    },
        "horcrux": {'Nagini': {"location": (3, 3),
                               "possible_locations": ((2, 2), (3, 3), (1, 1)),
                               "prob_change_location": 0.5}
                    },
        "death_eaters": {'Lucius Malfoy': {"index": 0,
                                           "path": [(1, 1), (1, 0)]},
                        'Snape': {"index": 0,
                                "path": [(5, 4), (5, 5), (5, 4)]}},
                        'random_de': {"index": 0,
                                "path": [(3, 3)]}
    },

    

]