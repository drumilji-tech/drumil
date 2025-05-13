import pandas as pd
import urllib
from sqlalchemy import create_engine
from Utils.Enums import DataSourceType
from Model.Turbine import Turbine


class RepositoryFactory:
    """A factory for creating data repositories."""

    def create_repository(
        data_source_type=DataSourceType.MS_SQL_DATABASE,
        data_file_path=None,
        data=None,
        data_file_index=0,
        data_file_parse_dates=True,
    ):
        """Creates a new data repository.

        Args:
            data_source_type (DataSourceType): The type of data source to use.
            data_file_path (str): The path to the CSV file to use as the data source (if `data_source_type` is `DataSourceType.CSV`).
            data_file_index (int): The column to use as the index of the data (if `data_source_type` is `DataSourceType.CSV`).
            data_file_parse_dates (bool): Whether to parse dates in the CSV file (if `data_source_type` is `DataSourceType.CSV`).

        Returns:
            A new data repository.
        """
        repo = None
        if data_source_type == DataSourceType.MS_SQL_DATABASE:
            repo = MSSQL_Repository()

        if data_source_type == DataSourceType.CSV:
            repo = CSV_Repository(
                data_file_path=data_file_path,
                data=data,
                data_file_index=data_file_index,
                data_file_parse_dates=data_file_parse_dates,
            )
        return repo


class CSV_Repository:
    """A repository that reads data from a CSV file."""

    def __init__(
        self,
        data_file_path=None,
        data=None,
        data_file_index=0,
        data_file_parse_dates=True,
        freq=None,
    ):
        """Initializes a new instance of the CSV_Repository class.

        Args:
            data_file_path (str): The path to the CSV file to use as the data source.
            data (pandas.DataFrame): can pass in a dataframe instead of a path.
            data_file_index (int): The column to use as the index of the data.
            data_file_parse_dates (bool): Whether to parse dates in the CSV file.
        """
        self._path = data_file_path
        self._data_file_index = data_file_index
        self._data_file_parse_dates = data_file_parse_dates
        self._data = data
        self._main_freq = freq if freq is not None else "10T"
        self._dataframes = []

        if data is not None:
            self._dataframes.append(data)

    @property
    def data(self):
        """Gets the data from the CSV file.

        Returns:
            pandas.DataFrame: The data from the CSV file.
        """
        if self._data is None:
            self._data = self.get_data()

        return self._data

    def get_data(self):
        """Reads the data from the CSV file.

        Returns:
            pandas.DataFrame: The data from the CSV file.
        """
        if self._path:
            data = pd.read_csv(
                self._path,
                index_col=self._data_file_index,
                parse_dates=self._data_file_parse_dates,
            )

            return data
        else:
            return None

    def add_data(self, new_data, freq=None):
        """Adds new data to the repository."""

        new_freq = freq if freq is not None else pd.infer_freq(new_data.index)

        if new_freq is None:
            raise ValueError("Frequency of new dataset could not be inferred")

        # If the main dataframe is None, assign new_data to it
        if self.data is None:
            self._data = new_data
            return
        main_freq = self._main_freq

        if main_freq is None:
            raise ValueError("Frequency of main dataset could not be inferred")

        if main_freq == new_freq:
            # Check for overlapping columns
            overlapping_columns = set(self._data.columns).intersection(
                set(new_data.columns)
            )

            # If columns are mutually exclusive and dates overlap
            if not overlapping_columns and any(self._data.index.isin(new_data.index)):
                self._data = pd.concat([self._data, new_data], axis=1)
            else:
                self._data = pd.concat([self._data, new_data])
        else:
            # If frequencies don't match, append to _dataframes without overwriting main dataframe
            self._dataframes.append(new_data)

    def get_column_data(self, column_names, freq=None):
        """Retrieve data for specific columns.

        Args:
            column_names (list or str): The name(s) of the column(s) to
                retrieve.
            freq (str, optional): The frequency of the data to
                retrieve. If not provided and a column exists in multiple
                dataframes, an error is raised.

        Returns:
            pd.DataFrame: A dataframe containing the requested column data.

        Raises:
            ValueError: If any specified column is not found or if the frequency
            is ambiguous.
        """

        # Ensure column_names is a list
        if isinstance(column_names, str):
            column_names = [column_names]

        dfs_to_concat = []
        for col in column_names:
            if col in self.data.columns:
                dfs_to_concat.append(self.data[[col]])
                continue

            # If not found in main dataframe, check in the additional dataframes
            found_dataframes = [df for df in self._dataframes if col in df.columns]

            if len(found_dataframes) == 0:
                raise ValueError(f"Column '{col}' not found in any dataframe.")

            elif len(found_dataframes) > 1 and freq is None:
                raise ValueError(
                    f"Column '{col}' exists in multiple dataframes. Please specify a frequency."
                )

            for df in found_dataframes:
                if pd.infer_freq(df.index) == freq or (
                    freq is None and pd.infer_freq(df.index) is not None
                ):
                    dfs_to_concat.append(df[[col]])
                    break
            else:
                raise ValueError(
                    f"Column '{col}' not found for the specified frequency."
                )

        result_df = pd.concat(dfs_to_concat, axis=1)
        return result_df

    def get_all_column_names(self):
        """Returns all unique column names across the repository."""

        # Start with columns from the main dataframe
        all_columns = set(self.data.columns)

        # Add columns from each additional dataframe in _dataframes
        for df in self._dataframes:
            all_columns.update(list(set(df.columns)))

        return sorted(list(all_columns))


class MSSQL_Repository:
    """A repository that reads data from a Microsoft SQL Server database."""

    def __init__(
        self,
        server="",
        database="",
        uid="",
        pwd="",
    ):
        self._server = server
        self._database = database
        self._UID = uid
        self._pwd = pwd
        self._engine = None

    def connect(self):
        """Connects to the SQL Server and returns a connection object.

        Returns:
            sqlalchemy.engine.Connection: A connection to the SQL Server.
        """
        if self._engine is None:
            if all(
                x is not None
                for x in [self._server, self._database, self._UID, self._pwd]
            ):
                params = urllib.parse.quote_plus(
                    r"DRIVER={ODBC Driver 17 for SQL Server}"
                    + r";SERVER={};DATABASE={};UID={};PWD={}".format(
                        self._server, self._database, self._UID, self._pwd
                    )
                )
                conn_str = "mssql+pyodbc:///?odbc_connect={}".format(params)
                self._engine = create_engine(conn_str)
            else:
                raise ValueError(
                    "MSSQLRepository: must provide server, database, uid, and pwd to connect to the SQL server"
                )

        connection = self._engine.connect()
        return connection
