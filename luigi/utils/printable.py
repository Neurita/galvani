# -*- coding: utf-8 -*-


class Printable(object):
    """

    """
    def __init__(self):
        pass

    def __str__(self):
        sb = []
        for key in self.__dict__:
            sb.append("{key}='{value}'".format(key=key,
                                               value=self.__dict__[key]))

        return ', '.join(sb)

    def __repr__(self):
        return self.__str__()
