# coding=utf-8
import codecs
import logging
import gzip
import json
import numpy as np
import os
from lib.ct import ct
from gensim import models

path_fb2m = '../data/fb2m/freebase-FB2M.txt'
import lib.data_helper as data_helper


class freebase:
    # 列出FB_2M 中所有提及的 entity 和 relation 输出到 data/fb2m
    # www.freebase.com/m/018fj69	www.freebase.com/music/recording/artist	www.freebase.com/m/01wbgdv
    @staticmethod
    def excat_fbxm(file_name='../data/fb2m/freebase-FB2M.txt', path="../data/simple_questions"):
        f1_writer = codecs.open("%s/e.txt" % path, mode="w", encoding="utf-8")
        f2_writer = codecs.open("%s/r.txt" % path, mode="w", encoding="utf-8")
        e1 = []
        r1 = []
        index = 0
        with codecs.open(file_name, mode="r", encoding="utf-8") as read_file:
            for line in read_file.readlines():
                index += 1
                if index % 10000 == 0:
                    print("==============", index)
                count = 0
                try:
                    count = len(line.split('\t'))
                except Exception as e2:
                    print(line)
                    print(index)
                    print(e2)
                    return

                # if count != 3:
                #     print(count)
                #     continue
                line = line.replace("www.freebase.com/", "").replace("\n", "")
                # f1_writer.write(line.split('\t')[0])
                e1.append(line.split('\t')[0])
                e2 = line.split('\t')[2]
                e2_s = e2.split(' ')
                if len(e2_s) > 0:
                    for e2_s1 in e2_s:
                        e1.append(e2_s1)
                else:
                    e1.append(e2)
                r1.append(line.split('\t')[1])
                # print("==========")

        for _ in e1:
            f1_writer.write(_ + "\n")
        for _ in r1:
            f2_writer.write(_ + "\n")

        f1_writer.close()
        f2_writer.close()

    @staticmethod
    def filter_raw_one(f_index):
        path1 = r'D:\ZAIZHI\freebase-data-full'
        rdf_length = len('<http://rdf.freebase.com/ns/')
        r_filter_set = set(('dataworld.gardening_hint.last_referenced_by', 'dataworld.gardening_hint.replaced_by',
                            'type.content.media_type', 'type.content.length', 'type.content.uploaded_by',
                            'measurement_unit.dated_integer.number', 'measurement_unit.dated_integer.year',
                            'tv.tv_series_episode.air_date', 'tv.tv_series_episode.season_number',
                            'media_common.netflix_title.netflix_genres', 'measurement_unit.rect_size.y',
                            'measurement_unit.rect_size.x', "type.object.type", 'book.book_edition.lcc',
                            'book.book_edition.LCCN', 'common.topic.topical_webpage', 'music.release.catalog_number',
                            'business.consumer_product.gtin', 'common.topic.official_website', 'type.object.key',
                            'common.document.text', 'type.content.blob_id', 'freebase.object_hints.best_hrid',
                            'common.document.source_uri', 'common.document.updated', 'common.document.text',
                            'common.topic.description', 'common.topic.topic_equivalent_webpage', 'common.topic.image',
                            'common.resource.annotations', 'type.object.permission', 'type.permission.controls',
                            'common.topic.webpage', 'book.book_edition.ISBN', 'common.licensed_object.license',
                            'tv.tv_program.thetvdb_id', 'type.object.id', 'type.type.instance', 'music.single.versions',
                            'music.recording.canonical_version', 'music.performance_role.track_performances',
                            'music.track_contribution.role', 'people.profession.people_with_this_profession',
                            'media_common.creative_work.credit', 'food.beer_country_region.beers_from_here',
                            'biology.breed_coloring.breeds', 'biology.organism_classification.organisms_of_this_type',
                            'military.military_unit_size.units_of_this_type',
                            'astronomy.type_of_planetographic_feature.planetographic_features_of_this_type',
                            'exhibitions.type_of_exhibition.exhibitions_of_this_type',
                            'time.time_zone.locations_in_this_time_zone', 'cvg.gameplay_mode.games_with_this_mode',
                            'fictional_universe.character_gender.characters_of_this_gender', 'biology.genome.gene',
                            'sports.sport.teams', 'people.cause_of_death.people', 'sports.sports_position.players',
                            'film.personal_film_appearance_type.film_appearances',
                            'film.special_film_performance_type.film_performance_type',
                            'astronomy.asteroid_group.asteroid_group_members',
                            'medicine.drug_formulation_category.drug_formulations',
                            'biology.organism_classification_rank.organism_classifications',
                            'olympics.olympic_games.athletes', 'film.content_rating.film', 'sports.sport.pro_athletes',
                            'people.ethnicity.people', 'education.educational_degree.people_with_this_degree',
                            'business.job_title.people_with_this_title',
                            'people.marriage_union_type.unions_of_this_type',
                            'tv.tv_producer_type.tv_producers_of_this_type',
                            'tv.special_tv_performance_type.starring_performances',
                            'business.issue_type.issues_of_this_type', 'business.company_type.companies_of_this_type',
                            'wine.wine_type.wines', 'geography.lake_type.lakes_of_this_type',
                            'metropolitan_transit.transit_service_type.transit_lines',
                            'geography.mountain_type.mountains_of_this_type',
                            'tv.special_tv_performance_type.episode_performances',
                            'cvg.computer_game_performance_type.performances',
                            'tv.tv_producer_type.episodes_with_this_role',
                            'digicams.camera_storage_type.compatible_cameras',
                            'transportation.bridge_type.bridges_of_this_type',
                            'organization.organization_type.organizations_of_this_type',
                            'award.competition_type.competitions_of_this_type',
                            'digicams.camera_sensor_type.digital_cameras', 'amusement_parks.ride_type.rides',
                            'travel.accommodation_type.accommodation_of_this_type',
                            'aviation.aircraft_type.aircraft_of_this_type', 'internet.top_level_domain_type.domains',
                            'event.disaster_type.disasters_of_this_type',
                            'aviation.accident_type.aircraft_accidents_of_this_type',
                            'metropolitan_transit.transit_service_type.transit_systems',
                            'astronomy.celestial_object_category.objects',
                            'education.school_category.schools_of_this_kind',
                            'government.government_office_category.officeholders',
                            'protected_sites.site_listing_category.listed_sites', 'award.award_category.winners',
                            'geography.geographical_feature_category.features',
                            'medicine.drug_pregnancy_category.drugs_in_this_category', 'award.award_category.nominees',
                            'interests.collection_category.items_in_this_category',
                            'protected_sites.iucn_category.protected_areas',
                            'government.government_office_category.offices', 'internet.website_category.sites',
                            'engineering.engine_category.engines', 'visual_art.visual_art_form.artists',
                            'visual_art.visual_art_form.artworks',
                            'medicine.drug_dosage_form.formulations_available_in_this_form',
                            'digicams.camera_iso.cameras', 'time.event.includes_event',
                            'medicine.disease.includes_diseases', 'people.person.quotations',
                            'people.cause_of_death.includes_causes_of_death',
                            'tv.non_character_role.tv_regular_personal_appearances',
                            'theater.theater_designer_role.designers',
                            'tv.non_character_role.tv_guest_personal_appearances', 'projects.project_role.projects',
                            'wine.wine_sub_region.wines',
                            'fictional_universe.character_powers.characters_with_this_ability',
                            'digicams.camera_compressed_format.cameras', 'film.film_format.film_format',
                            'broadcast.radio_format.stations', 'book.periodical_format.periodicals_in_this_format',
                            'digicams.camera_format.cameras', 'computer.computing_platform.file_formats_supported',
                            'wine.wine_color.wines', 'wine.wine_region.wines', 'wine.grape_variety.wines',
                            'book.magazine.issues', 'wine.appellation.wines', 'location.location.people_born_here',
                            'location.country.second_level_divisions', 'film.film_location.featured_in_films',
                            'location.location.events', 'location.country.administrative_divisions',
                            'location.place_with_neighborhoods.neighborhoods', 'location.us_county.hud_county_place',
                            'book.book_subject.works', 'film.film_subject.films',
                            'media_common.quotation_subject.quotations_about_this_subject',
                            'cvg.computer_game_subject.games', 'visual_art.art_subject.artwork_on_the_subject',
                            'book.periodical_subject.periodicals', 'influence.influence_node.influenced',
                            'film.film_genre.films_in_this_genre', 'media_common.literary_genre.books_in_this_genre',
                            'media_common.netflix_genre.titles', 'cvg.cvg_genre.games', 'tv.tv_genre.programs',
                            'theater.theater_genre.plays_in_this_genre',
                            'media_common.literary_genre.stories_in_this_genre',
                            'book.magazine_genre.magazines_in_this_genre', 'visual_art.visual_art_genre.artworks',
                            'games.game_genre.boardgames', 'opera.opera_genre.operas_in_this_genre',
                            'computer.software_genre.software_in_genre'))
        with gzip.open('../data/freebase_full/fb_raw/%d.gz' % (f_index,), 'wb') as f_out:
            with gzip.open('%s\%d.gz' % (path1, f_index,), 'rb') as f_in:
                for l in f_in:
                    l_list = l.decode('utf-8').strip().split('\t')
                    if l_list[1].startswith('<http://www.w3.org/') or l_list[2].startswith("\"/wikipedia/"):
                        continue
                    if l_list[1].startswith('<http://rdf.freebase.'):
                        r = l_list[1][rdf_length:-1]
                    else:
                        r = l_list[1]
                    if r.startswith('/') or r.startswith('user.') or r.endswith('openlibrary_id') or r.startswith(
                            'media_common.cataloged_instance.') or r.startswith('/authority.') or r.endswith(
                        '.uri') or r.startswith('freebase.user_activity') or r.endswith('.isbn'):
                        continue
                    if (r.startswith('music.') and (not r.endswith('instruments_played')) and (
                            not r.startswith('music.artist.')) and (r != 'music.musical_group.member') and (
                                r != 'music.composer.compositions')):
                        continue
                    if (r.startswith('type.content') or r.startswith('media_common.') or r.startswith(
                            'base.') or r.startswith('freebase.') or r.startswith('common.image') or r.startswith(
                        'freebase.valuenotation') or r.endswith('_id') or r.startswith(
                        'common.webpage') or r.startswith('base.schemastaging')):
                        continue
                    if (r == 'common.notable_for.display_name' or r == 'type.object.name' or r == 'common.topic.alias') \
                            and (not l_list[2].endswith('@en')):
                        continue
                    if r in r_filter_set:
                        continue
                    if l_list[0].startswith('<http://rdf.freebase.'):
                        arg1 = l_list[0][rdf_length:-1]
                    else:
                        arg1 = l_list[0]
                    if arg1.startswith('g.') or arg1.startswith('<http') or arg1.startswith('\"/rabj'):
                        continue
                    if l_list[2].startswith('<http://rdf.freebase.'):
                        arg2 = l_list[2][rdf_length:-1]
                    else:
                        arg2 = l_list[2]
                    if arg2.startswith('g.') or arg2.startswith('<http'):
                        continue
                    for date_str in ('#gYearMonth>', '#date>', '#gYear>'):
                        if arg2.endswith(date_str):
                            arg2 = '%s^^%s' % (arg2.split('^^')[0], date_str[1:-1])
                    # print >> f_out, '\t'.join((arg1, r, arg2)).encode('utf-8')
                    msg = ('\t'.join((arg1, r, arg2)) + "\n").encode('utf-8')
                    f_out.write(msg)

    # 把拆开的314份文件，合并成30+份文件
    @staticmethod
    def split_fb_raw_data(f_in_index_list):
        """

        :param f_in_index_list:range(start,end)
        :return:
        """
        i = 0
        f_out_index = 0
        with gzip.open('../data/freebase_full/fb_combine/%d.gz' % (f_out_index,), 'wb') as f_out:
            for f_in_index in f_in_index_list:
                print("%d / %d " % (f_in_index, len(f_in_index_list)))
                with gzip.open('../data/freebase_full/fb_raw/%d.gz' % (f_in_index,), 'rb') as f_in:
                    for l in f_in:
                        i += 1
                        f_out.write(l)
                        if i >= 10000000:
                            i = 0
                            f_out_index += 1
                            f_out = gzip.open('../data/freebase_full/fb_combine/%d.gz' % (f_out_index,), 'wb')
                            #
                            # 输出剩下的

    # 从freebase中抽取实体的相关信息到一个单独的文件
    @staticmethod
    def excat_entity_rdf_from_fb(e_path, out_path, out_path2, is_record):
        # e_path = '../data/fb2m/e.txt'
        print("excat_entity_rdf_from_fb")
        # 1 读取所有entity (e1)
        e_set = ct.file_read_all_lines(e_path)
        print("init e_lines %d" % e_set.__len__())
        e_set = set([str(x).replace("\r", "").replace("\n", "").replace("m/", "m.") for x in e_set])
        e_dict = dict()
        i = 0
        print("init e_set")
        r_lear_set = set()

        # 2 挨个读取freebase文件
        for f_index in range(0, 24):  # 24
            with gzip.open('../data/freebase_full/fb_combine/%d.gz' % (f_index,), 'rb') as f_in:
                for l in f_in:
                    i += 1
                    if i % 100000 == 0:
                        print("%d - 2400 " % (i / 100000))
                    l_list = l.decode('utf-8').strip().split('\t')
                    e1 = l_list[0]
                    # 3 逐行判断是否RDF中 s 是 e1
                    if e1 in e_set:
                        # 4 如果在其中则将其添加到e1的压缩包中
                        r_e2 = (l_list[1].replace("\r", "").replace("\n", ""),
                                l_list[2].replace("\r", "").replace("\n", ""))
                        if e1 in e_dict:
                            e_v1 = e_dict[e1]
                            e_v1.append(r_e2)
                            e_dict[e1] = e_v1
                        else:
                            # e_v1 = []
                            # e_v1.append(r_e2)
                            e_dict[e1] = [r_e2]
        # 4 输出所有
        del e_set  # 减少内存
        print("start output")
        i = 0
        total = 0

        for e, e_v in e_dict.items():
            i += 1
            if i % 10000 == 0:
                print("%s / %s " % (i / 10000, total))
                ct.print_t()
            # print("%s "%(e,))
            for r_e2 in e_v:
                r_lear_set.add(r_e2[0])

            # is_record = False
            if not is_record:
                continue
            with gzip.open('%s/%s.gz' % (out_path, e), 'wb') as f_out:
                for r_e2 in e_v:
                    # print("%s-%s"%(r_e2[0],r_e2[1]))
                    f_out.write(("%s\t%s\n" % (r_e2[0], r_e2[1])).encode('utf-8'))

        for r in r_lear_set:
            ct.just_log('%s/freebase_relation_clear.txt' % (out_path2), r)
        print(432432423)

    @staticmethod
    def excat_annotated_fb_data(num):
        fname = "../data/simple_questions/annotated_fb_data_all.txt"
        lines = ct.file_read_all_lines(fname)
        if num > 0:
            lines = lines[0:num]
        with codecs.open("%s-%d.txt" % (fname, num), mode="a",
                         encoding="utf-8") as f1_writer:
            for l in lines:
                f1_writer.write(l)

    # 从标注的问题中提取出对应的rdf和e1
    @staticmethod
    def excat_entity_in_annotated_fb_data(fname="../data/simple_questions/annotated_fb_data_train-1000.txt"):
        lines = ct.file_read_all_lines(fname)
        e1_set = set()
        for l in lines:
            l = str(l).replace("www.freebase.com/", "")
            e1 = "m." + l.split("\t")[0][2:]
            e2 = "m." + l.split("\t")[2][2:]
            e1_set.add(e1)
            e1_set.add(e2)
        for e in e1_set:
            ct.just_log(fname + "-entity.txt", e)
        print(1)

    @staticmethod
    def prodeuce_embedding_vec_file(filename):
        dh = data_helper.DataClass("sq")
        model = models.Word2Vec.load(filename)
        # 遍历每个单词，查出word2vec然后输出

        v_base = model['end']
        ct.print(v_base)

        for word in dh.converter.vocab:
            try:
                v = model[word]
            except Exception as e1:
                msg1 = "%s : %s " % (word, e1)
                ct.print(msg1)
                ct.just_log("../data/simple_questions/fb_0_files/wiki.vector.log", msg1)
                v = model['end']
            m_v = ' '.join([str(x) for x in list(v)])
            msg = "%s %s" % (word, str(m_v))
            # ct.print(msg)
            ct.just_log("../data/simple_questions/fb_0_files/wiki.vector", msg)
        m_v = ' '.join([str(x) for x in list(v)])
        msg = "%s %s" % (word, str(m_v))
        msg = "%s %s" % ('end', msg)
        ct.just_log("../data/simple_questions/fb_0_files/wiki.vector", msg)


if __name__ == "__main__":
    # s1 :
    # freebase.excat_annotated_fb_data(0)

    # s2
    # freebase.excat_fbxm(file_name="../data/simple_questions/annotated_fb_data_all.txt-0.txt",
    #          path="../data/simple_questions/fb_0_files")
    # s3
    # p1 = "../data/simple_questions/fb_0_files/e.txt"
    # # # p1 = "../data/simple_questions/annotated_fb_data_all.txt-0.txt"
    # out_path = '../data/simple_questions/fb_0'
    # out_path2 = '../data/simple_questions/fb_0_files'
    # freebase.excat_entity_rdf_from_fb(p1, out_path, out_path2, is_record=True)

    # s4 :  freebase.excat_entity_in_annotated_fb_data

    # filename1 = '../data/word2vec/train.model.1516630487.7132027'
    # filename2 = '../data/word2vec/wiki.vector'
    # freebase.prodeuce_embedding_vec_file(filename1) #  生成wiki.vector文件

    # for i in range(314):
    # freebase.split_fb_raw_data(range(314))
    # freebase.filter_raw_one(i)
    # print("%d ok" % i)
    print(1)
