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
        "wizards": {'Harry Potter': {"location": (2, 0)}},
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
        "wizards": {'Harry Potter': {"location": (2, 2)}},
        "horcrux": {
            'Nagini': {"location": (0, 2),
                       "possible_locations": ((0, 2), (1, 2), (2, 2)),
                       "prob_change_location": 0.9},
            'Diary': {"location": (0, 0),
                      "possible_locations": ((0, 0), (1, 0), (2, 0)),
                      "prob_change_location": 0.3}
        },
        "death_eaters": {
            'Snape': {"index": 1, "path": [(2, 2), (2, 1)]}
        },
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
        "wizards": {'Harry Potter': {"location": (0, 2)}},
        "horcrux": {
            'Nagini': {"location": (0, 2),
                       "possible_locations": ((0, 2), (1, 2), (2, 2)),
                       "prob_change_location": 0.9},
            'Diary': {"location": (0, 0),
                      "possible_locations": ((0, 0), (2, 0)),
                      "prob_change_location": 0.45}
        },
        "death_eaters": {
            'Snape': {"index": 1, "path": [(1, 0), (2, 1)]},
            'random_de': {"index": 0, "path": [(3, 2)]}
        },
    },

    {
        "optimal": True,
        "turns_to_go": 20,
        "map": [
            ['P','P','I','P'],
            ['P','P','P','P'],
            ['I','P','P','P'],
            ['P','P','P','I']
        ],
        "wizards": {'Harry Potter': {"location": (1,0)}},
        "horcrux": {
            'Nagini': {"location": (0,0),
                       "possible_locations": ((0,0),(1,1),(3,2)),
                       "prob_change_location": 0.3}
        },
        "death_eaters": {
            'Lucius': {"index": 0, "path": [(1,2),(1,3)]}
        }
    },
    {
        "optimal": True,
        "turns_to_go": 15,
        "map": [
            ['P','P','P'],
            ['P','I','P'],
            ['P','P','P'],
        ],
        "wizards": {'Harry Potter': {"location": (2,2)}},
        "horcrux": {
            'Nagini': {"location": (0,0),
                       "possible_locations": ((0,0),(2,2)),
                       "prob_change_location": 0.6}
        },
        "death_eaters": {
            'De1': {"index": 0, "path": [(1,0),(1,1),(1,2)]}
        }
    },
    {
        "optimal": True,
        "turns_to_go": 50,
        "map": [
            ['P','P','P','P'],
            ['P','P','P','P'],
            ['P','P','I','P'],
            ['P','P','P','P']
        ],
        "wizards": {'Harry': {"location": (0,0)}},
        "horcrux": {
            'Ring': {"location": (3,3),
                     "possible_locations": ((3,3),(2,1),(1,2)),
                     "prob_change_location": 0.2}
        },
        "death_eaters": {
            'Draco': {"index": 0, "path": [(1,1),(2,1)]}
        }
    },
    {
        "optimal": True,
        "turns_to_go": 25,
        "map": [
            ['P','P','I','P'],
            ['I','P','P','P'],
            ['P','P','P','P']
        ],
        "wizards": {'Harry': {"location": (2,0)}},
        "horcrux": {
            'Cup': {"location": (0,3),
                    "possible_locations": ((0,3),(1,2),(2,3)),
                    "prob_change_location": 0.1}
        },
        "death_eaters": {
            'RandomDE': {"index": 0, "path": [(1,0)]}
        }
    },
    {
        "optimal": True,
        "turns_to_go": 40,
        "map": [
            ['P','P','P','P','P'],
            ['P','I','P','I','P'],
            ['P','P','P','P','P']
        ],
        "wizards": {'Harry': {"location": (2,2)}},
        "horcrux": {
            'Locket': {"location": (0,4),
                       "possible_locations": ((0,4),(1,2)),
                       "prob_change_location": 0.7}
        },
        "death_eaters": {
            'DE_1': {"index": 0, "path": [(0,1),(1,1)]}
        }
    },
    {
        "optimal": True,
        "turns_to_go": 10,
        "map": [
            ['P','I','P','P'],
            ['P','P','P','I'],
            ['P','P','P','P']
        ],
        "wizards": {'Harry': {"location": (0,0)}},
        "horcrux": {
            'Nagini': {"location": (2,3),
                       "possible_locations": ((2,3),(1,2)),
                       "prob_change_location": 0.5}
        },
        "death_eaters": {
            'Goyle': {"index":0,"path":[(1,0),(1,1)]}
        }
    },
    {
        "optimal": True,
        "turns_to_go": 60,
        "map": [
            ['P','P','P'],
            ['P','P','P'],
            ['P','P','P'],
            ['I','I','P']
        ],
        "wizards": {'Harry': {"location": (0,0)}},
        "horcrux": {
            'Diadem': {"location": (2,2),
                       "possible_locations": ((2,2),(1,1)),
                       "prob_change_location": 0.8}
        },
        "death_eaters": {
            'DE_Short': {"index":0,"path":[(1,2)]}
        }
    },
    {
        "optimal": True,
        "turns_to_go": 12,
        "map": [
            ['P','P','P','P'],
            ['P','I','P','P'],
            ['P','P','P','P'],
        ],
        "wizards": {'Harry': {"location": (2,3)}},
        "horcrux": {
            'Hor1': {"location": (0,0),
                     "possible_locations": ((0,0),(2,1)),
                     "prob_change_location": 0.4}
        },
        "death_eaters": {
            'DE_A': {"index":0,"path":[(1,0),(1,1)]}
        }
    },
    {
        "optimal": True,
        "turns_to_go": 18,
        "map": [
            ['P','P','I'],
            ['P','P','P'],
            ['P','P','P'],
        ],
        "wizards": {'Harry': {"location": (1,1)}},
        "horcrux": {
            'Hor2': {"location": (2,2),
                     "possible_locations": ((2,2),(1,2)),
                     "prob_change_location": 0.35}
        },
        "death_eaters": {
            'DE_B': {"index":0,"path":[(2,0),(2,1)]}
        }
    },
    {
        "optimal": True,
        "turns_to_go": 80,
        "map": [
            ['P','P','P','P'],
            ['P','P','I','P'],
            ['P','P','P','P'],
            ['P','P','P','P']
        ],
        "wizards": {'Harry': {"location": (3,3)}},
        "horcrux": {
            'HorcruxA': {"location": (0,0),
                         "possible_locations": ((0,0),(2,2)),
                         "prob_change_location": 0.75}
        },
        "death_eaters": {
            'DE_C': {"index":0,"path":[(1,1),(1,2)]}
        }
    },
    {
        "optimal": True,
        "turns_to_go": 90,
        "map": [
            ['P','P','P','P','P'],
            ['P','I','P','I','P'],
            ['P','P','P','P','P'],
            ['P','P','P','P','P']
        ],
        "wizards": {'Harry': {"location": (0,0)}},
        "horcrux": {
            'Hrc1': {"location": (3,4),
                     "possible_locations": ((3,4),(2,2),(0,4)),
                     "prob_change_location": 0.3}
        },
        "death_eaters": {
            'DE_10': {"index":0,"path":[(1,0),(2,0)]}
        }
    },
    {
        "optimal": True,
        "turns_to_go": 15,
        "map": [
            ['P','P','P'],
            ['P','P','P'],
            ['I','I','P']
        ],
        "wizards": {'Harry': {"location": (0,2)}},
        "horcrux": {
            'Hrc2': {"location": (2,2),
                     "possible_locations": ((2,2),(0,1)),
                     "prob_change_location": 0.5}
        },
        "death_eaters": {
            'DE_12': {"index":0,"path":[(1,2)]}
        }
    },
    {
        "optimal": True,
        "turns_to_go": 22,
        "map": [
            ['P','P','P','I'],
            ['P','P','P','P'],
            ['P','P','I','P']
        ],
        "wizards": {'Harry': {"location": (1,1)}},
        "horcrux": {
            'Hrc3': {"location": (0,0),
                     "possible_locations": ((0,0),(1,2),(2,3)),
                     "prob_change_location": 0.65}
        },
        "death_eaters": {
            'DE_13': {"index":0,"path":[(2,0),(2,1)]}
        }
    },
    {
        "optimal": True,
        "turns_to_go": 35,
        "map": [
            ['P','P','P'],
            ['P','P','P'],
            ['P','P','P']
        ],
        "wizards": {'Harry': {"location": (2,2)}},
        "horcrux": {
            'Hrc4': {"location": (0,0),
                     "possible_locations": ((0,0),(2,1),(2,2)),
                     "prob_change_location": 0.85}
        },
        "death_eaters": {
            'DE_14': {"index":0,"path":[(1,0),(1,1),(1,2)]}
        }
    },
    {
        "optimal": True,
        "turns_to_go": 45,
        "map": [
            ['P','P','P','P'],
            ['P','I','P','P'],
            ['P','P','P','P'],
            ['P','P','P','P']
        ],
        "wizards": {'Harry': {"location": (0,0)}},
        "horcrux": {
            'Hrc5': {"location": (3,3),
                     "possible_locations": ((3,3),(2,1)),
                     "prob_change_location": 0.4}
        },
        "death_eaters": {
            'DE_15': {"index":0,"path":[(1,2),(2,2)]}
        }
    },
    {
        "optimal": True,
        "turns_to_go": 28,
        "map": [
            ['P','P','P'],
            ['P','I','P'],
            ['P','P','P'],
        ],
        "wizards": {'Harry': {"location": (0,0)}},
        "horcrux": {
            'Hrc6': {"location": (2,2),
                     "possible_locations": ((2,2),(0,2)),
                     "prob_change_location": 0.5}
        },
        "death_eaters": {
            'DE_16': {"index":0,"path":[(1,0),(1,1),(1,2)]}
        }
    },
    {
        "optimal": True,
        "turns_to_go": 38,
        "map": [
            ['P','P','P','P'],
            ['P','P','P','I'],
            ['P','P','P','P']
        ],
        "wizards": {'Harry': {"location": (2,0)}},
        "horcrux": {
            'Hrc7': {"location": (0,3),
                     "possible_locations": ((0,3),(2,2)),
                     "prob_change_location": 0.9}
        },
        "death_eaters": {
            'DE_17': {"index":0,"path":[(1,1)]}
        }
    },
    {
        "optimal": True,
        "turns_to_go": 70,
        "map": [
            ['P','P','P'],
            ['P','P','P'],
            ['P','I','P'],
            ['P','P','P']
        ],
        "wizards": {'Harry': {"location": (0,2)}},
        "horcrux": {
            'Hrc8': {"location": (3,2),
                     "possible_locations": ((3,2),(2,2)),
                     "prob_change_location": 0.15}
        },
        "death_eaters": {
            'DE_18': {"index":0,"path":[(1,0),(1,1),(1,2)]}
        }
    },
    {
        "optimal": True,
        "turns_to_go": 95,
        "map": [
            ['P','P','P','P'],
            ['P','P','P','P'],
            ['P','I','P','I'],
            ['P','P','P','P']
        ],
        "wizards": {'Harry': {"location": (0,0)}},
        "horcrux": {
            'Hrc9': {"location": (3,3),
                     "possible_locations": ((3,3),(1,3)),
                     "prob_change_location": 0.25}
        },
        "death_eaters": {
            'DE_19': {"index":0,"path":[(2,0),(2,1)]}
        }
    },
    {
        "optimal": True,
        "turns_to_go": 55,
        "map": [
            ['P','P','P'],
            ['P','P','P'],
            ['P','P','P']
        ],
        "wizards": {'Harry': {"location": (2,0)}},
        "horcrux": {
            'Hrc10': {"location": (0,2),
                      "possible_locations": ((0,2),(1,1),(2,2)),
                      "prob_change_location": 0.33}
        },
        "death_eaters": {
            'DE_20': {"index":0,"path":[(1,0),(1,1),(1,2)]}
        }
    },

    # סה"כ 3 (ישנים) + 20 (חדשים) = 23 עם optimal=True,
    # אבל ביקשנו 25. נוסיף עוד 2:

    {
        "optimal": True,
        "turns_to_go": 30,
        "map": [
            ['P','P','P','I'],
            ['P','P','P','P'],
            ['P','P','I','P'],
            ['P','P','P','P'],
        ],
        "wizards": {'Harry': {"location": (3,0)}},
        "horcrux": {
            'Gem': {"location": (0,0),
                    "possible_locations": ((0,0),(2,3)),
                    "prob_change_location": 0.55}
        },
        "death_eaters": {
            'DE_Gem': {"index":0,"path":[(1,0),(1,1)]}
        }
    },
    {
        "optimal": True,
        "turns_to_go": 10,
        "map": [
            ['P','P','I','P'],
            ['P','P','I','P'],
            ['P','P','P','P']
        ],
        "wizards": {'Harry': {"location": (2,3)}},
        "horcrux": {
            'TomRiddleBook': {"location": (0,0),
                              "possible_locations": ((0,0),(1,2),(2,2)),
                              "prob_change_location": 0.88}
        },
        "death_eaters": {
            'DE_Book': {"index":0,"path":[(1,3)]}
        }
    },

]
