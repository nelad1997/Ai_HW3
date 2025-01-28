inputs = [
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

    {
        "optimal": False,
        "turns_to_go": 20,
        "map": [
            ['P','P','I','P'],
            ['I','P','P','P'],
            ['P','P','P','P']
        ],
        "wizards": {'Harry': {"location": (2,0)}},
        "horcrux": {
            'HorX': {"location": (0,3),
                     "possible_locations": ((0,3),(1,2),(2,3)),
                     "prob_change_location": 0.2}
        },
        "death_eaters": {
            'DE_X': {"index":0,"path":[(1,1),(1,0)]}
        }
    },
    {
        "optimal": False,
        "turns_to_go": 12,
        "map": [
            ['P','I','P'],
            ['P','P','P'],
            ['P','P','P']
        ],
        "wizards": {'Harry': {"location": (0,0)}},
        "horcrux": {
            'Harcrux': {"location": (2,2),
                        "possible_locations": ((2,2),(0,2)),
                        "prob_change_location": 0.3}
        },
        "death_eaters": {
            'DE_X2': {"index":0,"path":[(1,1)]}
        }
    },
    {
        "optimal": False,
        "turns_to_go": 40,
        "map": [
            ['P','P','P','P'],
            ['P','P','P','P'],
            ['I','P','P','P'],
            ['P','P','P','P']
        ],
        "wizards": {
            'Harry': {"location": (3,0)},
            'Ron':   {"location": (2,1)}
        },
        "horcrux": {
            'Cup2': {"location": (0,3),
                     "possible_locations": ((0,3),(1,2),(2,3)),
                     "prob_change_location": 0.1}
        },
        "death_eaters": {
            'DE_X3': {"index":0,"path":[(1,0),(1,1)]}
        }
    },
    {
        "optimal": False,
        "turns_to_go": 35,
        "map": [
            ['P','P','P'],
            ['P','P','I'],
            ['P','P','P']
        ],
        "wizards": {'Harry': {"location": (2,2)}},
        "horcrux": {
            'Lock2': {"location": (0,0),
                      "possible_locations": ((0,0),(2,0)),
                      "prob_change_location": 0.6}
        },
        "death_eaters": {
            'DE_X4': {"index":0,"path":[(1,2)]}
        }
    },
    {
        "optimal": False,
        "turns_to_go": 25,
        "map": [
            ['P','P','P','P'],
            ['P','I','P','I'],
            ['P','P','P','P']
        ],
        "wizards": {'Harry': {"location": (0,0)}},
        "horcrux": {
            'Lock3': {"location": (2,3),
                      "possible_locations": ((2,3),(1,2)),
                      "prob_change_location": 0.4}
        },
        "death_eaters": {
            'DE_X5': {"index":0,"path":[(1,0)]}
        }
    },
    {
        "optimal": False,
        "turns_to_go": 30,
        "map": [
            ['P','P','P'],
            ['P','I','P'],
            ['P','P','P'],
            ['P','P','P']
        ],
        "wizards": {'Harry': {"location": (3,2)}},
        "horcrux": {
            'Lock4': {"location": (0,0),
                      "possible_locations": ((0,0),(2,2)),
                      "prob_change_location": 0.5}
        },
        "death_eaters": {
            'DE_X6': {"index":0,"path":[(1,2)]}
        }
    },
    {
        "optimal": False,
        "turns_to_go": 45,
        "map": [
            ['P','P','P','P'],
            ['P','I','P','P'],
            ['P','P','P','P'],
            ['P','P','P','P']
        ],
        "wizards": {'Harry': {"location": (0,0)}},
        "horcrux": {
            'Hxx': {"location": (3,3),
                    "possible_locations": ((3,3),(2,1)),
                    "prob_change_location": 0.8}
        },
        "death_eaters": {
            'DE_Ba': {"index":0,"path":[(1,2),(2,2)]}
        }
    },
    {
        "optimal": False,
        "turns_to_go": 55,
        "map": [
            ['P','P','P'],
            ['P','P','P'],
            ['P','P','P']
        ],
        "wizards": {'Harry': {"location": (2,0)}},
        "horcrux": {
            'Hyy': {"location": (0,2),
                    "possible_locations": ((0,2),(2,2)),
                    "prob_change_location": 0.9}
        },
        "death_eaters": {
            'DE_Bb': {"index":0,"path":[(1,0),(1,1)]}
        }
    },
    {
        "optimal": False,
        "turns_to_go": 22,
        "map": [
            ['P','I','P'],
            ['P','P','P'],
            ['P','P','I']
        ],
        "wizards": {'Harry': {"location": (2,0)}},
        "horcrux": {
            'Hzz': {"location": (0,2),
                    "possible_locations": ((0,2),(1,2)),
                    "prob_change_location": 0.25}
        },
        "death_eaters": {
            'DE_Bc': {"index":0,"path":[(1,0),(1,1)]}
        }
    },
    {
        "optimal": False,
        "turns_to_go": 18,
        "map": [
            ['P','P','P','P'],
            ['P','P','I','P'],
            ['P','P','P','P'],
        ],
        "wizards": {'Harry': {"location": (2,3)}},
        "horcrux": {
            'HrB': {"location": (0,0),
                    "possible_locations": ((0,0),(2,1)),
                    "prob_change_location": 0.5}
        },
        "death_eaters": {
            'DE_Cx': {"index":0,"path":[(1,1)]}
        }
    },
    {
        "optimal": False,
        "turns_to_go": 28,
        "map": [
            ['P','P','I'],
            ['P','P','P'],
            ['P','P','P'],
        ],
        "wizards": {'Harry': {"location": (1,1)}},
        "horcrux": {
            'HrD': {"location": (2,2),
                    "possible_locations": ((2,2),(1,2)),
                    "prob_change_location": 0.65}
        },
        "death_eaters": {
            'DE_Cy': {"index":0,"path":[(2,0),(2,1)]}
        }
    },
    {
        "optimal": False,
        "turns_to_go": 66,
        "map": [
            ['P','P','P','P'],
            ['P','P','I','P'],
            ['P','P','P','P'],
            ['P','P','P','P']
        ],
        "wizards": {'Harry': {"location": (3,3)}},
        "horcrux": {
            'HrE': {"location": (0,0),
                    "possible_locations": ((0,0),(2,2)),
                    "prob_change_location": 0.45}
        },
        "death_eaters": {
            'DE_Cz': {"index":0,"path":[(1,1),(1,2)]}
        }
    },
    {
        "optimal": False,
        "turns_to_go": 75,
        "map": [
            ['P','P','P','P','P'],
            ['P','I','P','I','P'],
            ['P','P','P','P','P'],
            ['P','P','P','P','P']
        ],
        "wizards": {'Harry': {"location": (0,0)}},
        "horcrux": {
            'HrF': {"location": (3,4),
                    "possible_locations": ((3,4),(2,2),(0,4)),
                    "prob_change_location": 0.2}
        },
        "death_eaters": {
            'DE_Cz2': {"index":0,"path":[(1,0),(2,0)]}
        }
    },
    {
        "optimal": False,
        "turns_to_go": 33,
        "map": [
            ['P','P','P'],
            ['P','P','P'],
            ['I','I','P']
        ],
        "wizards": {'Harry': {"location": (0,2)}},
        "horcrux": {
            'HrG': {"location": (2,2),
                    "possible_locations": ((2,2),(0,1)),
                    "prob_change_location": 0.22}
        },
        "death_eaters": {
            'DE_Sm': {"index":0,"path":[(1,2)]}
        }
    },
    {
        "optimal": False,
        "turns_to_go": 37,
        "map": [
            ['P','P','P','I'],
            ['P','P','P','P'],
            ['P','P','I','P']
        ],
        "wizards": {'Harry': {"location": (1,1)}},
        "horcrux": {
            'HrH': {"location": (0,0),
                    "possible_locations": ((0,0),(1,2),(2,3)),
                    "prob_change_location": 0.72}
        },
        "death_eaters": {
            'DE_Sn': {"index":0,"path":[(2,0),(2,1)]}
        }
    },
    {
        "optimal": False,
        "turns_to_go": 44,
        "map": [
            ['P','P','P'],
            ['P','P','P'],
            ['P','P','P']
        ],
        "wizards": {'Harry': {"location": (2,2)}},
        "horcrux": {
            'HrI': {"location": (0,0),
                    "possible_locations": ((0,0),(2,1),(2,2)),
                    "prob_change_location": 0.28}
        },
        "death_eaters": {
            'DE_So': {"index":0,"path":[(1,0),(1,1),(1,2)]}
        }
    },
    {
        "optimal": False,
        "turns_to_go": 59,
        "map": [
            ['P','P','P','P'],
            ['P','I','P','I'],
            ['P','P','P','P']
        ],
        "wizards": {'Harry': {"location": (0,0)}},
        "horcrux": {
            'HrJ': {"location": (2,3),
                    "possible_locations": ((2,3),(1,2)),
                    "prob_change_location": 0.36}
        },
        "death_eaters": {
            'DE_Sp': {"index":0,"path":[(1,0)]}
        }
    },
    {
        "optimal": False,
        "turns_to_go": 29,
        "map": [
            ['P','P','P'],
            ['P','I','P'],
            ['P','P','P'],
            ['P','P','P']
        ],
        "wizards": {'Harry': {"location": (3,2)}},
        "horcrux": {
            'HrK': {"location": (0,0),
                    "possible_locations": ((0,0),(2,2)),
                    "prob_change_location": 0.7}
        },
        "death_eaters": {
            'DE_Sq': {"index":0,"path":[(1,2)]}
        }
    },
    {
        "optimal": False,
        "turns_to_go": 11,
        "map": [
            ['P','P','P','P'],
            ['P','P','I','P'],
            ['P','P','P','P']
        ],
        "wizards": {'Harry': {"location": (2,3)}},
        "horcrux": {
            'HrL': {"location": (0,0),
                    "possible_locations": ((0,0),(2,1)),
                    "prob_change_location": 0.46}
        },
        "death_eaters": {
            'DE_Sr': {"index":0,"path":[(1,1)]}
        }
    },
    {
        "optimal": False,
        "turns_to_go": 19,
        "map": [
            ['P','P','I'],
            ['P','P','P'],
            ['P','P','P'],
        ],
        "wizards": {'Harry': {"location": (1,1)}},
        "horcrux": {
            'HrM': {"location": (2,2),
                    "possible_locations": ((2,2),(1,2)),
                    "prob_change_location": 0.55}
        },
        "death_eaters": {
            'DE_Ss': {"index":0,"path":[(2,0),(2,1)]}
        }
    },

    {
        "optimal": False,
        "turns_to_go": 28,
        "map": [
            ['P','P','P','P','P'],
            ['P','I','P','P','P'],
            ['P','P','P','I','P'],
            ['P','P','P','P','P']
        ],
        "wizards": {
            'Harry': {"location": (0,0)},
            'Ron':   {"location": (3,4)}
        },
        "horcrux": {
            'HorA': {"location": (2,4),
                     "possible_locations": ((2,4),(1,2),(3,3)),
                     "prob_change_location": 0.4},
        },
        "death_eaters": {
            'DE_X21': {"index":0,"path":[(1,0),(1,1)]},
            'DE_X22': {"index":0,"path":[(2,2)]}
        }
    },

    {
        "optimal": False,
        "turns_to_go": 35,
        "map": [
            ['P','P','P','P'],
            ['P','P','P','P'],
            ['P','I','P','P'],
            ['P','P','P','P']
        ],
        "wizards": {
            'Harry': {"location": (3,0)},
            'Hermione': {"location": (0,3)}
        },
        "horcrux": {
            'HorB': {"location": (2,3),
                     "possible_locations": ((2,3),(1,3)),
                     "prob_change_location": 0.25},
            'HorC': {"location": (1,0),
                     "possible_locations": ((1,0),(2,0)),
                     "prob_change_location": 0.6}
        },
        "death_eaters": {
            'DE_X23': {"index":0,"path":[(2,1),(1,1)]}
        }
    },

    {
        "optimal": False,
        "turns_to_go": 24,
        "map": [
            ['P','P','I'],
            ['P','P','P'],
            ['P','P','P'],
            ['P','I','P']
        ],
        "wizards": {
            'Harry': {"location": (0,0)},
        },
        "horcrux": {
            'HorD': {"location": (3,2),
                     "possible_locations": ((3,2),(2,2),(1,2)),
                     "prob_change_location": 0.45}
        },
        "death_eaters": {
            'DE_X24': {"index":0,"path":[(2,0),(2,1),(2,2)]},
            'DE_X25': {"index":0,"path":[(1,0)]}
        }
    }

]