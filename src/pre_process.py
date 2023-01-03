from datetime import datetime

from pandas import (
    DataFrame, to_datetime, options, get_dummies,
    concat
)

from sklearn.model_selection import train_test_split

#TODO: Config management, column names, types, transformations should be stored in configs.

options.mode.chained_assignment = None  

class CleanData:
    """The class prepares data for analysis and feature transformation"""
    def __init__(self, df: DataFrame):
        self.orig_df = df

    def remove_nulls(self, df: DataFrame) -> DataFrame:
        return df[df['POL_STATUS'].notnull()]

    def adjust_date_types(self, df: DataFrame) -> DataFrame:
        # US date format
        df['QUOTE_DATE'] = to_datetime(df['QUOTE_DATE'], format=r"%m/%d/%Y")
        
        # UK date format
        df['COVER_START'] = to_datetime(df['COVER_START'], format=r"%d/%m/%Y")
        df['P1_DOB'] = to_datetime(df['P1_DOB'], format=r"%d/%m/%Y")

        return df

    def adjust_categorical_types(self, df: DataFrame) -> DataFrame:

        categorical_columns = [
            'P1_EMP_STATUS', 'P1_PT_EMP_STATUS', 'BUS_USE',
            'NEIGH_WATCH', 'PAYMENT_METHOD', 'LEGAL_ADDON_POST_REN',
            'SUBSIDENCE', 'P1_MAR_STATUS', 
            'P1_MAR_STATUS', 'KEYCARE_ADDON_PRE_REN', 'HP1_ADDON_POST_REN',
            'SEC_DISC_REQ', 'CLAIM3YEARS', 'AD_CONTENTS', 'AD_BUILDINGS', 
            'MTA_FLAG', 'SAFE_INSTALLED', 'P1_POLICY_REFUSED', 'HP3_ADDON_PRE_REN',
            'HP2_ADDON_POST_REN', 'HP3_ADDON_POST_REN', 'HOME_EM_ADDON_PRE_REN',
            'BUILDINGS_COVER', 'APPR_ALARM', 'CONTENTS_COVER', 'OCC_STATUS', 'FLOODING',
            'GARDEN_ADDON_PRE_REN', 'HP2_ADDON_PRE_REN', 'FLOODING', 'APPR_LOCKS', 
            'LEGAL_ADDON_PRE_REN'
        ]

        for col in categorical_columns:
            df[col] = df[col].astype("category")

        return df


    def clean(self) -> DataFrame:
        print("Cleaning data...")
        clean_df = (
            self.orig_df
                .pipe(self.remove_nulls)
                .pipe(self.adjust_date_types)
                .pipe(self.adjust_categorical_types)
        )

        return clean_df

class PreProcess:
    """The class prepares performs feature transformation"""
    def __init__(self, df: DataFrame) -> DataFrame:
        self.orig_df = df

    def existing_customer(self, df: DataFrame) -> DataFrame:
        df['EXISTING_CUSTOMER'] =  ((df['COVER_START'] - df['QUOTE_DATE']).dt.days < 0).astype(int)
        return df

    def month_of_year(self, df: DataFrame) -> DataFrame:
        df['QUOTE_MONTH'] = df['QUOTE_DATE'].dt.month
        return df

    def customer_age(self, df: DataFrame) -> DataFrame:
        df['AGE_YEARS'] = df['QUOTE_DATE'].dt.year - df['P1_DOB'].dt.year
        return df

    def handle_field_level_nulls(self, df: DataFrame) -> DataFrame:
        non_cat_strategies = {
            'RISK_RATED_AREA_B': -1,
            'RISK_RATED_AREA_C': -1,
            'PAYMENT_FREQUENCY': -1
        }

        df.fillna(non_cat_strategies, inplace=True)

        return df

    def make_binary_flags(self, df: DataFrame) -> DataFrame:
        y_n_cols = [
            'LEGAL_ADDON_PRE_REN', 'CONTENTS_COVER',
            'LEGAL_ADDON_POST_REN', 'HP2_ADDON_PRE_REN',
            'P1_POLICY_REFUSED', 'BUS_USE', 'AD_BUILDINGS',
            'APPR_LOCKS', 'HP1_ADDON_PRE_REN', 'NEIGH_WATCH',
            'HP2_ADDON_POST_REN', 'HP3_ADDON_PRE_REN', 
            'GARDEN_ADDON_POST_REN', 'SEC_DISC_REQ', 
            'CLAIM3YEARS', 'AD_CONTENTS', 'HP3_ADDON_POST_REN',
            'MTA_FLAG', 'APPR_ALARM', 'KEYCARE_ADDON_PRE_REN',
            'GARDEN_ADDON_PRE_REN', 'FLOODING', 'BUILDINGS_COVER',
            'SAFE_INSTALLED', 'SUBSIDENCE', 'HOME_EM_ADDON_POST_REN', 
            'KEYCARE_ADDON_POST_REN', 'HP1_ADDON_POST_REN',
            'HOME_EM_ADDON_PRE_REN'
        ]

        replacements = {'Y': 1, 'N': 0}
        
        for col in y_n_cols:
            df[col] = df[col].replace(to_replace=replacements).astype(int)
        
        return df

    def encode_and_bind(self, df: DataFrame, column: str) -> DataFrame:
        dummies = get_dummies(
            df[[column]],
            prefix=f"{column}_"
        )
        
        result = concat([df, dummies], axis=1)
        
        return result.drop(column, axis=1)

    def one_hot_encoding(self, df: DataFrame) -> DataFrame:
        encoded_df = (
            df
            .pipe(self.encode_and_bind, 'P1_PT_EMP_STATUS')
            .pipe(self.encode_and_bind, 'P1_MAR_STATUS')
            .pipe(self.encode_and_bind, 'P1_EMP_STATUS')
            .pipe(self.encode_and_bind, 'OCC_STATUS')
            .pipe(self.encode_and_bind, 'PAYMENT_METHOD')
        )

        return encoded_df

    def transform_target(self, df: DataFrame) -> DataFrame:
        df['LAPSED'] = (df['POL_STATUS'] == 'Lapsed').astype(int)
        return df

    def remove_unneeded_columns(self, df: DataFrame) -> DataFrame:

        df = (
            df
            .drop('CLERICAL', axis=1)
            .drop('MTA_DATE', axis=1)
            .drop('POL_STATUS', axis=1)
            .drop('P1_DOB', axis=1)
            .drop('P1_SEX', axis=1)
            .drop('COVER_START', axis=1) 
            .drop('Police', axis=1)
            .drop('i', axis=1)         
        )

        return df

    def split_holdout(self, df: DataFrame, cut_dttm: datetime) -> list:
        train_test =  (
            df[df['QUOTE_DATE'] < cut_dttm]
            .drop('QUOTE_DATE', axis=1)
        )

        holdout = (
            df[df['QUOTE_DATE'] >= cut_dttm]
            .drop('QUOTE_DATE', axis=1)
        )

        return train_test, holdout

    def important_features(self, df: DataFrame, enable=False) -> DataFrame:
        features = [
            'AD_BUILDINGS', 
            'SUM_INSURED_BUILDINGS', 
            'LEGAL_ADDON_POST_REN', 
            'KEYCARE_ADDON_PRE_REN',
            'HP1_ADDON_POST_REN',
            'HP2_ADDON_POST_REN', 
            'HP3_ADDON_POST_REN',
            'OCC_STATUS__LP', 
            'PAYMENT_METHOD__PureDD',
            'RISK_RATED_AREA_B',
            'NCD_GRANTED_YEARS_B',
            'NCD_GRANTED_YEARS_C',
            'RISK_RATED_AREA_C',
            'AGE_YEARS',
            'YEARBUILT',
            'SPEC_ITEM_PREM',
            'P1_MAR_STATUS__M',
            'QUOTE_DATE',
            'LAPSED'
        ]
        if enable:
            return df[features]
        else:
            return df
    
    def process(self, important_features=False) -> tuple:
        print("Pre-processing data...")
        processed_df = (
            self.orig_df
                .pipe(self.existing_customer)
                .pipe(self.customer_age)
                .pipe(self.handle_field_level_nulls)
                .pipe(self.month_of_year)
                .pipe(self.make_binary_flags)
                .pipe(self.one_hot_encoding)
                .pipe(self.transform_target) 
                .pipe(self.remove_unneeded_columns) 
                .pipe (self.important_features, enable=important_features)
        )

        train_test, val = self.split_holdout(
            processed_df, datetime(2010, 1, 4, 0, 0, 0)
        )

        X_train_test = train_test.loc[:, train_test.columns != 'LAPSED']
        y_train_test = train_test[['LAPSED']]

        X_val = val.loc[:, val.columns != 'LAPSED']
        y_val = val[['LAPSED']]

        X_train, X_test, y_train, y_test = train_test_split(
           X_train_test, y_train_test, 
           test_size=0.33, random_state=42
        )

        return X_train, X_test, X_val, y_train, y_test, y_val
