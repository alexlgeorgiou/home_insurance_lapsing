import pandas
import pytest 
import src.pre_process as pp

#TODO: Add comprehensive coverage for tests

def test_adjust_date_types_pass():

    mock_data = {
        'QUOTE_DATE':['12/22/2010'], 
        'COVER_START':['22/12/2010'],
        'P1_DOB':['22/12/2010'],
    }
    mock_df = pandas.DataFrame(mock_data)

    pp.CleanData(mock_df).adjust_date_types(mock_df)


def test_adjust_date_types_fail():
    with pytest.raises(Exception):
        mock_data = {
            'QUOTE_DATE':['12/100/2010'], 
            'COVER_START':['22/22/2010'],
            'P1_DOB':['22/22/2010'],
        }
        mock_df = pandas.DataFrame(mock_data)

        pp.CleanData(mock_df).adjust_date_types(mock_df)
