#!/usr/bin/python


import requests
import json
import random

API_KEY = "LIVDSRZULELA"
API_ENDPOINT = 'https://api.tenor.co/v1/'
DEFAULT_SEARCH_LIMIT = 25
SAFE_SEARCH = 'off'

class TenorApiException(Exception):
    pass

class TenorImage(object):

    def __init__(self, data = None):
        if data:
            self.created = data.get('created')
            self.url = data.get('url')
            self.tags = data.get('tags')
            self.type = data.get('tupe')
            self.dims = ""
            self.preview = ""
            self.size = ""
            self.duration = ""


class Tenor(object):
    """
    Tenor API Documentation
    https://api.tenor.co/#start
    This object can grab gifs from a service via a rest interface
    Example REST call:
    https://api.tenor.co/v1/search?tag=hello&key=VSJGLF2U743Y
    """
    def __init__(self):
        self.api_key = API_KEY
        #self.weburl = data.get('weburl')
        #self.results = data.get('results')

    def _get(self, **params):
        """
        API Request wrapper
        :param params:  List of available params found at the Tenor API Documentation linked above, under the Search heading.
        :return:  JSON-encoded data representing a number of gifs and gif metadata
        """
        params['api_key'] = self.api_key

        response = requests.get('https://api.tenor.co/v1/search', params=params)

        results = json.loads(response.text)

        return results

    def search(self, tag, safesearch = None, limit = None):
        """
        :param tag:  a tag or search string
        :param safesearch: (values:off|moderate|strict) specify the content safety filter level
        :param limit: fetch up to a specified number of results (max: 50).
        :return:  JSON-encoded data representing a number of gifs and gif metadata
        """

        params = {'tag': tag}

        if safesearch:
            params['safesearch'] = safesearch

        if limit:
            params['limit'] = limit

        results = self._get(**params)

        return results

    def random(self, tag):

        search_results = self.search(tag=tag)

        random_entry = random.choice(search_results['results'])
        gif = random_entry['media'][0]['gif']['url']

        return gif
    
    def randommp4(self, tag):

        search_results = self.search(tag=tag)

        random_entry = random.choice(search_results['results'])
        gif = random_entry['media'][0]['loopedmp4']['url']

        return gif