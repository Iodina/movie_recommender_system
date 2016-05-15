#!/usr/bin/env python
# -*- coding: utf-8 -*-
import re
import urllib2
from urllib2 import Request
from bs4 import BeautifulSoup

__author__ = 'mishashamen'


def cut_substring_between(first, last):
    def decorator(get_string):

        def wrapped(*func_args):
            try:
                s = get_string(*func_args)
                start = s.index(first) + len(first)
                end = s.index(last, start)
                return s[start:end]
            except ValueError:
                return ""

        return wrapped

    return decorator


class Film:
    def __init__(self,
                 parser,
                 id,
                 title=None,
                 year=None,
                 country=None,
                 director=None,
                 genre=None,
                 time=None,
                 mean_rating=None):
        self.__parser = parser
        self.__id = id
        self.__title = title
        self.__year = year
        self.__country = country
        self.__director = director
        self.__genre = genre
        self.__time = time
        self.__mean_rating = mean_rating

    def lazy_load(self):
        self.__parser.get_film_by_id(self.__id)

    def get_id(self):
        return self.__id

    def get_title(self):
        if self.__title is None:
            self.lazy_load()
        return self.__title

    def set_title(self, title):
        self.__title = title

    def get_year(self):
        if self.__year is None:
            self.lazy_load()
        return self.__year

    def set_year(self, year):
        self.__year = year

    def get_country(self):
        if self.__country is None:
            self.lazy_load()
        return self.__country

    def set_country(self, country):
        self.__country = country

    def get_director(self):
        if self.__director is None:
            self.lazy_load()
        return self.__director

    def set_director(self, director):
        self.__director = director

    def get_genre(self):
        if self.__genre is None:
            self.lazy_load()
        return self.__genre

    def set_genre(self, genre):
        self.__genre = genre

    def get_time(self):
        if self.__time is None:
            self.lazy_load()
        return self.__time

    def set_time(self, time):
        self.__time = time

    def get_mean_rating(self):
        if self.__mean_rating is None:
            self.lazy_load()
        return self.__mean_rating

    def set_mean_rating(self, mean_rating):
        self.__mean_rating = mean_rating


class User:
    def __init__(self, id):
        self.__id = id
        self.__films = []

    def get_id(self):
        return self.__id

    def get_films(self):
        return self.__films

    def add_film(self, film_id, rate):
        self.get_films().append({film_id: rate})


class RequestSender:
    ban_request_message = '<script src="http://www.google.com/recaptcha/'
    proxies = ['http://www.google.ie/gwt/x?u=',
               'http://www.google.ua/gwt/x?u=',
               'http://www.google.com/gwt/x?u=']

    def get_html_page(self, url):
        for proxy in self.proxies:
            req = Request(proxy + url,
                          data=None,
                          headers={
                              'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_9_3) '
                                            'AppleWebKit/537.36 (KHTML, like Gecko) '
                                            'Chrome/35.0.1916.47 Safari/537.36'}
                          )
            try:
                page = urllib2.urlopen(req).read()
                if not self.is_bad_page(page):
                    return page
            except Exception as e:
                print e

        return ""

    def is_bad_page(self, page):
        if self.ban_request_message in page:
            return True

        return False


class Parser:
    search_query = "http://www.kinopoisk.ru/s/type/all/find/"
    film_profile_query = "http://www.kinopoisk.ru/film/"

    def __init__(self, rs):
        self.rs = rs
        self.property_map = {
            u'год': self.set_film_year,
            u'страна': self.set_film_country,
            u'режиссер': self.set_film_director,
            u'жанр': self.set_film_genre,
            u'время': self.set_film_time}

    def get_film_with_id_by_title(self, title):
        search_film_url = self.search_query + title
        film_id = self.get_film_id_on_search_page(search_film_url)
        return Film(self, film_id)

    @cut_substring_between("/film/", "/")
    def get_film_id_on_search_page(self, search_film_url):
        soup = BeautifulSoup(self.rs.get_html_page(search_film_url),
                             'html.parser')
        for info in soup.find('div', {'class': 'search_results'}).find_all('div', {'class': 'info'}):
            for name in info.find_all("p", {'class': 'name'}):
                return name.a.get('href')

    def find_film_page_by_id(self, film_id):
        film_url = self.film_profile_query + film_id
        return self.rs.get_html_page(film_url)

    def get_film_users(self, film):
        users = []
        film_id = film.get_id()
        # Todo
        return users

    def get_film_by_id(self, film_id):
        film = Film(self, film_id)
        film_page = self.find_film_page_by_id(film_id)
        soup = BeautifulSoup(film_page, 'html.parser')
        self.set_film_title(film, soup)
        self.set_film_mean_rating(film, soup)
        for tr in soup.table.find_all('tr'):
            td_key, td_value = tr.find_all('td')
            setter = self.property_map.get(td_key.text)
            if setter is not None:
                setter(film, td_value)
        return film

    def set_film_title(self, film, soup):
        film.title = soup.h1.text

    def set_film_year(self, film, cell_content):
        film.set_year(int(cell_content.text));

    def set_film_country(self, film, cell_content):
        country_id = int(re.search('(?<=/)\d+',
                                   cell_content.a.get("href")).group())

        country_name = "".join(cell_content.a.text.split())
        film.set_country({country_id: country_name})

    def set_film_director(self, film, cell_content):
        director_id = int(re.search('(?<=/)\d+',
                                cell_content.a.get("href")).group())
        director_name = ' '.join(cell_content.a.text.split())
        film.set_director({director_id: director_name})

    def set_film_genre(self, film, cell_content):
        genre_id = int(re.search('(?<=/)\d+',
                             cell_content.a.get("href")).group())
        genre_name = ''.join(cell_content.a.text.split())
        film.set_genre({genre_id: genre_name})

    def set_film_time(self, film, cell_content):
        time = int(re.search('\d+ ',
                         cell_content.text).group(0))
        film.set_time(time)

    def set_film_mean_rating(self, film, soup):
        rate = float(soup.find('span', {'class': 'rating_ball'}).text)
        count = int(''.join(soup.find('span', {'class': 'ratingCount'}).text.split()))
        film.set_mean_rating({'rate': rate, 'count': count})


film_id = "4871"

p = Parser(RequestSender())

film = p.get_film_by_id(film_id)

print p.get_film_id_on_search_page("Запах женщины")
