import polars as pl


# Ingest csv from CareLink
class CarelinkIngest:
    def __init__(self, file_path):
        """
        Initialize the carelink class and automatically read the CSV file
        :param file_path: file path pointing to CSV of downloaded CareLink data
        """
        # Read csv's and separate into df's
        raw = [self.__read(fp) for fp in file_path]
        # Combine df's of the same type (by location)
        self.combined_data = []
        for i in range(len(raw[0])):
            meta = []
            data = []
            for file_i in range(len(raw)):
                if raw[file_i][i]['meta'] is not None:
                    meta = list(set(meta + raw[file_i][i]['meta']))
                data = [pl.concat(data + [raw[file_i][i]['data']])]
            meta = None if not meta else meta  # convert empty list to None
            data = data[0]  # unlist data
            data = data.unique()  # dedup data
            self.combined_data += [{'meta': meta, 'data': data}]
        # Call __split() to split data and file/table metadata
        self.data, self.table_metadata, self.file_metadata = self.__split(self.combined_data)

    def __read(self, file_path):
        """
        Read carelink data csv and reformat into multiple tables
        :param file_path: file path pointing to CSV of downloaded CareLink data
        :return: list of dicts; each dict contains a "meta" key containing a list of metadata values
                 and a "data" key containing a Polars df of a table from the input file path
        """
        # Set the delim used in input csv
        delim = "-------"

        # Read the raw data
        data = pl.read_csv(file_path, has_header=False, schema={f'column_{i}': pl.String for i in range(1, 55)})

        # Remove null rows which space tables in data
        data = data.filter(pl.col("column_1").is_not_null())

        # Find indices where new tables start
        start_indices = (
                [0]  # add the first row of data
                + [i for i, val in enumerate(data["column_1"]) if val == delim]
                + [len(data)]  # add the last row of data
        )

        # Split into multiple DataFrames
        dfs = []
        for i in range(len(start_indices) - 1):
            start = start_indices[i]
            end = start_indices[i + 1]
            df = data[start:end]
            if i == 0:
                header_end = df.row(0).index('End Date')
                headers = df.row(0)[0:header_end + 1]
                values = df.row(1)[0:header_end + 1]
                meta1 = df.row(0)[header_end + 1:]
                meta2 = df.row(1)[header_end + 1:]
                for meta in [meta1, meta2]:
                    add_headers_i = [meta_i for meta_i in range(len(meta))
                                     if not meta_i % 2 and meta[meta_i] is not None]
                    headers = list(headers) + [meta[meta_i] for meta_i in add_headers_i]
                    values = list(values) + [meta[meta_i+1] for meta_i in add_headers_i]
                for ix in range(len(headers)):
                    df[0, ix] = headers[ix]
                    df[1, ix] = values[ix]
                df = df[0:2, :len(headers)]
                meta = None
            else:
                meta = [val for val in df.row(0) if val is not None and val.strip() != delim]
                df = df[1:, :]  # remove metadata  row
                df = df.select([col for col in df.columns if df[0, col] is not None])  # remove null column name cols
            df = df[1:, :].rename(lambda col_name: df.row(0)[df.columns.index(col_name)])
            dfs.append(
                {'meta': meta,
                 'data': df}
            )
        return dfs

    def __split(self, combined_data):
        """
        Split combined data list into list of Polars dataframes, and table/file metadata
        :param combined_data: list of dicts containing data (Polars dataframes) and meta (table metadata)
                              The first data table should be the file metadata
        :return: data: a list of Polars dataframes reprenting the input file tables,
                 table_metadata: a list of metadata corresponding to each table in `data`
                 file_metadata: a dict of named metadata which is relevant to the entire file
        """
        data = [combined_data[i]['data'] for i in range(1, len(combined_data))]
        table_metadata = [combined_data[i]['meta'] for i in range(1, len(combined_data))]
        meta = combined_data[0]['data']
        file_metadata = {}
        for key in meta.columns:
            val = ';'.join(meta[key].drop_nulls().unique())
            if val != '':
                file_metadata.update({key: val})
        return data, table_metadata, file_metadata

    def display_meta(self):
        """
        Print the overall and table-specific metadata for carelink_read outputs
        :return: None
        """
        to_print = '---- File metadata: ----\n'
        to_print += str(self.file_metadata)[1:-1].replace("', '", "',\n'") + '\n\n'
        for i in range(len(self.data)):
            to_print += f'---- Table {i} Metadata: ----\n'
            to_print += '\n'.join(self.table_metadata[i])
            to_print += '\nColumns: '
            to_print += str(self.data[i].columns) + '\n'
            to_print += str(self.data[i].schema) + '\n'
            if i != len(self.data) - 1:
                to_print += '\n'
        print(to_print)
