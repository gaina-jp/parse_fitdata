import warnings
import fitdecode
import pandas as pd
import streamlit as st
import numpy as np
import altair as alt

# fitdecodeの読み込み時に発生する型サイズの警告を無視する
warnings.filterwarnings('ignore', category=UserWarning, module='fitdecode')

def speed_to_pace_str(speed_ms):
    """m/s (秒速) を 1kmあたりのペース (M:S/km) に変換"""
    if pd.isna(speed_ms) or speed_ms is None or speed_ms <= 0.1:
        return None
    sec = 1000 / speed_ms
    minutes = int(sec // 60)
    seconds = int(sec % 60)
    return f"{minutes}:{seconds:02d}"

def pace_str_to_sec(pace_str):
    """'M:S' の文字列を秒数に変換（グラフのY軸計算用）"""
    if pd.isna(pace_str) or not isinstance(pace_str, str):
        return None
    try:
        m, s = pace_str.split(':')
        return int(m) * 60 + int(s)
    except Exception:
        return None

def semicircles_to_degrees(semicircles):
    """GarminのFITデータ特有のSemicircles単位を、一般的な緯度経度（Degrees）に変換する"""
    if pd.isna(semicircles) or semicircles is None:
        return None
    # 変換式: degrees = semicircles * ( 180 / 2^31 )
    return semicircles * (180.0 / (2**31))

def safe_mean(series, decimals=0):
    """NaNを安全に処理しながら平均値を計算し、丸めるヘルパー関数"""
    val = series.mean()
    if pd.isna(val):
        return None
    if decimals == 0:
        return int(round(val))
    return round(val, decimals)

def calculate_lap_splits(df):
    """1kmごとのラップデータを計算する"""
    if '距離' not in df.columns:
        return None

    df_calc = df.copy()

    # タイムスタンプ(インデックス)から時間の差分(秒)を計算
    df_calc['delta_time'] = df_calc.index.to_series().diff().dt.total_seconds()
    # 距離の差分(m)を計算
    df_calc['delta_dist'] = df_calc['距離'].diff()

    # 標高の差分から獲得標高と下降標高を計算
    if '標高' in df_calc.columns:
        df_calc['delta_alt'] = df_calc['標高'].diff()
        df_calc['gain'] = df_calc['delta_alt'].clip(lower=0)
        df_calc['loss'] = df_calc['delta_alt'].clip(upper=0).abs()
    else:
        df_calc['gain'] = 0
        df_calc['loss'] = 0

    # 1kmごとのラップ番号を算出 (距離はmなので // 1000)
    df_calc['lap'] = (df_calc['距離'] // 1000).fillna(0).astype(int) + 1

    splits = []
    # ラップ番号ごとにデータを集計
    for lap, group in df_calc.groupby('lap'):
        split_data = {'ラップ': f"{lap} km"}

        # 距離の差分合計 (m) ※計算用に取得するが画面には表示しない
        split_dist = group['delta_dist'].sum()

        # ペースの平均 (そのラップ内の合計距離 / 合計時間)
        split_time = group['delta_time'].sum()
        if split_time > 0 and split_dist > 0:
            speed_ms = split_dist / split_time
            split_data['ペースの平均'] = speed_to_pace_str(speed_ms)
        else:
            split_data['ペースの平均'] = None

        if '心拍数' in group.columns:
            split_data['心拍数の平均'] = safe_mean(group['心拍数'], 0)

        if 'ストライド (cm)' in group.columns:
            split_data['ストライドの平均 (cm)'] = safe_mean(group['ストライド (cm)'], 1)

        if 'ケイデンス' in group.columns:
            split_data['ケイデンスの平均'] = safe_mean(group['ケイデンス'], 0)

        if '接地時間' in group.columns:
            split_data['接地時間の平均'] = safe_mean(group['接地時間'], 1)

        if '上下動' in group.columns:
            split_data['上下動の平均'] = safe_mean(group['上下動'], 1)

        if '垂直比' in group.columns:
            split_data['垂直比の平均'] = safe_mean(group['垂直比'], 1)

        split_data['獲得標高 (m)'] = round(group['gain'].sum(), 1)
        split_data['累積下降 (m)'] = round(group['loss'].sum(), 1)

        splits.append(split_data)

    df_splits = pd.DataFrame(splits)
    # データが存在しなかった(全てNoneの)列は削除
    df_splits = df_splits.dropna(axis=1, how='all')
    df_splits = df_splits.set_index('ラップ')
    return df_splits

def adjust_heart_rate_anomalies(df, threshold_bpm=10):
    """
    心拍数の欠落後に異常な値(小さすぎる値)が記録され、その後正常な値に戻る現象を補正する。
    また、値が欠損している区間（NaN）も、直前の値 (b) と復帰後の正常な値 (c) の間で補間する。
    """
    if 'heart_rate' not in df.columns:
        return df

    # NaNかどうかを判定するためのブール配列
    is_nan = df['heart_rate'].isna()
    
    # 欠損の開始位置と終了位置を見つけるためのグループ化
    # 欠損状態が変わるごとにgroupのIDが増加する
    nan_groups = (is_nan != is_nan.shift()).cumsum()
    
    # 補正済みのデータを格納するシリーズ
    corrected_hr = df['heart_rate'].copy()
    
    # 欠損しているグループのIDを取得
    missing_group_ids = nan_groups[is_nan].unique()
    
    for gid in missing_group_ids:
        missing_indices = df.index[nan_groups == gid]
        if len(missing_indices) == 0:
            continue
            
        first_missing_idx = missing_indices[0]
        last_missing_idx = missing_indices[-1]
        
        # 欠損の直前のインデックスを探す (b)
        loc_first_missing = df.index.get_loc(first_missing_idx)
        if loc_first_missing == 0:
            continue # 先頭が欠損の場合は無視
            
        b_idx = df.index[loc_first_missing - 1]
        b_val = df.at[b_idx, 'heart_rate']
        
        if pd.isna(b_val):
            continue # 直前も何らかの理由でNaNなら無視

        # 欠損の直後のインデックスを探す (a)
        loc_last_missing = df.index.get_loc(last_missing_idx)
        if loc_last_missing >= len(df) - 1:
            continue # 末尾が欠損の場合は無視
            
        a_idx = df.index[loc_last_missing + 1]
        a_val = df.at[a_idx, 'heart_rate']
        
        if pd.isna(a_val):
            continue

        # (a) が (b) より明らかに小さいかチェック
        if b_val - a_val >= threshold_bpm:
            # 異常な急降下があった場合
            # (c) を探す: a_idx より後で、値が b_val ± 2 になる最初のポイント
            c_idx = None
            c_val = None
            
            # a_idx の次から探索
            search_start_loc = loc_last_missing + 1
            
            for i in range(search_start_loc + 1, min(search_start_loc + 60, len(df))): # 最大60ポイント(約60秒)先まで探索
                current_idx = df.index[i]
                current_val = df.at[current_idx, 'heart_rate']
                
                if pd.isna(current_val):
                    continue
                    
                if abs(current_val - b_val) <= 2:
                    c_idx = current_idx
                    c_val = current_val
                    break
                    
            if c_idx is not None:
                # (b) の直後から (c) の直前までの補間を行う（NaNの部分も異常な急降下部分も含める）
                b_loc = df.index.get_loc(b_idx)
                c_loc = df.index.get_loc(c_idx)
                
                # 補間対象の要素数（両端 b, c を含まない間の要素数）
                num_points_to_interpolate = c_loc - b_loc - 1
                
                if num_points_to_interpolate > 0:
                    step = (c_val - b_val) / (num_points_to_interpolate + 1)
                    interpolated_values = [round(b_val + step * (i + 1)) for i in range(num_points_to_interpolate)]
                    
                    # b_idx の次 から c_idx の直前までを置き換え
                    for i, val in enumerate(interpolated_values):
                        idx_to_replace = df.index[b_loc + 1 + i]
                        corrected_hr.at[idx_to_replace] = val
        else:
            # 異常な急降下がなかった場合でも、NaNの区間だけを (b) と (a) の間で補間する
            b_loc = df.index.get_loc(b_idx)
            a_loc = df.index.get_loc(a_idx)
            
            num_points_to_interpolate = a_loc - b_loc - 1
            if num_points_to_interpolate > 0:
                step = (a_val - b_val) / (num_points_to_interpolate + 1)
                interpolated_values = [round(b_val + step * (i + 1)) for i in range(num_points_to_interpolate)]
                
                for i, val in enumerate(interpolated_values):
                    idx_to_replace = df.index[b_loc + 1 + i]
                    corrected_hr.at[idx_to_replace] = val

    df['heart_rate'] = corrected_hr
    return df

def parse_fit_file(uploaded_file):
    data = []
    
    with fitdecode.FitReader(uploaded_file) as fit_file:
        for frame in fit_file:
            if isinstance(frame, fitdecode.records.FitDataMessage) and frame.name == 'record':
                record_data = {}
                for field in frame.fields:
                    record_data[field.name] = field.value
                data.append(record_data)

    if not data:
        return None

    df = pd.DataFrame(data)

    # 英語の列名の中に空白が入っている場合（例: 'Effort Pace'）、
    # 扱いやすくするために空白をアンダースコアに変換し、小文字に統一する
    df.columns = [str(col).strip().replace(' ', '_').lower() for col in df.columns]

    # 不要な従来の 'altitude' と 'speed' を削除
    columns_to_drop = [col for col in ['altitude', 'speed'] if col in df.columns]
    if columns_to_drop:
        df = df.drop(columns=columns_to_drop)

    # enhanced_altitude を altitude にリネーム
    if 'enhanced_altitude' in df.columns:
        df = df.rename(columns={'enhanced_altitude': 'altitude'})

    # enhanced_speed を speed(M:S/km) に変換
    if 'enhanced_speed' in df.columns:
        df['speed'] = df['enhanced_speed'].apply(speed_to_pace_str)
        df = df.drop(columns=['enhanced_speed'])

    # effort_pace を文字列 (M:S/km) に変換
    if 'effort_pace' in df.columns:
        df['effort_pace'] = df['effort_pace'].apply(speed_to_pace_str)

    # 位置情報(Semicircles)を緯度経度(Degrees)に変換
    if 'position_lat' in df.columns:
        df['position_lat'] = df['position_lat'].apply(semicircles_to_degrees)
    if 'position_long' in df.columns:
        df['position_long'] = df['position_long'].apply(semicircles_to_degrees)

    # ストライド(step_length)を mm から cm に変換
    if 'step_length' in df.columns:
        df['step_length'] = df['step_length'] / 10.0

    # ケイデンスを両足の歩数(spm)にするため2倍に変換 (ランニングを想定)
    # 小数部分(fractional_cadence)が存在する場合は合算してから2倍にする
    if 'cadence' in df.columns:
        if 'fractional_cadence' in df.columns:
            df['cadence'] = (df['cadence'] + df['fractional_cadence']) * 2
        else:
            df['cadence'] = df['cadence'] * 2

    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # FITファイルのタイムスタンプは通常UTCなので、JST(日本標準時)に変換する
        if df['timestamp'].dt.tz is None:
            df['timestamp'] = df['timestamp'].dt.tz_localize('UTC').dt.tz_convert('Asia/Tokyo')
        else:
            df['timestamp'] = df['timestamp'].dt.tz_convert('Asia/Tokyo')
            
        df['timestamp'] = df['timestamp'].dt.tz_localize(None)

        df = df.set_index('timestamp')

    # --- 心拍数の異常値補正処理の追加 ---
    df = adjust_heart_rate_anomalies(df, threshold_bpm=15)

    # カラム名を日本語に変換
    # 'step_length': '歩幅' を 'ストライド (cm)' に変更
    column_translations = {
        'heart_rate': '心拍数',
        'altitude': '標高',
        'speed': 'ペース',
        'distance': '距離',
        'cadence': 'ケイデンス',
        'power': 'パワー',
        'temperature': '気温',
        'position_lat': '緯度',
        'position_long': '経度',
        'fractional_cadence': 'ケイデンス(小数)',
        'left_right_balance': '左右バランス',
        'gps_accuracy': 'GPS精度',
        'vertical_oscillation': '上下動',
        'stance_time_percent': '接地時間バランス',
        'stance_time': '接地時間',
        'activity_type': 'アクティビティタイプ',
        'step_length': 'ストライド (cm)',
        'effort_pace': 'エフォートペース',
        'vertical_ratio': '垂直比',
        'accumulated_power': '累積パワー'
    }
    df = df.rename(columns=column_translations)
    df.index.name = 'タイムスタンプ'

    # 指定されたカラムの順番を定義 (存在しない列は無視される)
    desired_order = [
        'アクティビティタイプ',
        '距離',
        '心拍数',
        'ペース',
        'エフォートペース',
        'ストライド (cm)',
        'ケイデンス',
        '接地時間',
        'パワー',
        '累積パワー',
        '上下動',
        '垂直比',
        '標高',
        '緯度',
        '経度'
    ]

    # DataFrameに存在する列のみを desired_order の順に並べる
    existing_columns = [col for col in desired_order if col in df.columns]

    # 指定したリストにない列 (気温など) があれば、その後ろにくっつける
    remaining_columns = [col for col in df.columns if col not in existing_columns]

    df = df[existing_columns + remaining_columns]

    return df

def main():
    st.set_page_config(page_title="FIT Data Analyzer", page_icon="🏃‍♂️", layout="wide")
    st.title("FIT Data Analyzer 🏃‍♂️🚴‍♀️")
    st.write("GarminなどのFITファイルをアップロードして、データを解析・グラフ化します。")

    uploaded_file = st.file_uploader("FITファイルをアップロードしてください", type=['fit'])

    if uploaded_file is not None:
        st.success(f"{uploaded_file.name} を読み込み中...")

        df = parse_fit_file(uploaded_file)

        if df is not None:
            st.subheader("データプレビュー")
            st.dataframe(df.head(10))

            # --- 1kmごとのラップデータを追加表示 ---
            lap_splits_df = calculate_lap_splits(df)
            if lap_splits_df is not None:
                st.subheader("1kmごとのラップデータ")
                # CSV出力には含めず、画面の表示のみ行う
                st.dataframe(lap_splits_df)

            st.subheader("グラフ化")

            # DataFrameの列から、数値データ（np.number）の列だけを初期候補として取得
            available_columns = df.select_dtypes(include=[np.number]).columns.tolist()

            # ペースとエフォートペースは上で文字列(M:S/km)に変換されてしまったため、
            # np.numberの判定からは漏れてしまう。そのため、手動でグラフの選択肢に追加する。
            if 'ペース' in df.columns:
                available_columns.append('ペース')
            if 'エフォートペース' in df.columns:
                available_columns.append('エフォートペース')

            # 選択肢の重複を排除してソート
            available_columns = list(set(available_columns))

            selected_columns = st.multiselect(
                "グラフに表示したいデータを選んでください",
                options=available_columns,
                default=[col for col in ['心拍数', '標高', 'ペース'] if col in available_columns]
            )

            if selected_columns:
                plot_df_base = df.reset_index()
                x_col = 'タイムスタンプ' if 'タイムスタンプ' in plot_df_base.columns else 'index'
                x_type = 'T' if x_col == 'タイムスタンプ' else 'Q'

                dfs_to_concat = []
                for col in selected_columns:
                    temp_df = pd.DataFrame()
                    temp_df['Time'] = plot_df_base[x_col]
                    temp_df['Metric'] = col

                    # ペース、エフォートペースの場合は、グラフのY軸に描画するために
                    # 「5:30」のような文字列を「330(秒)」という数値に変換して Value にセットする。
                    # DisplayValue には元の文字列を残し、ツールチップ(マウスオーバー)で見せる。
                    if col in ['ペース', 'エフォートペース']:
                        temp_df['Value'] = plot_df_base[col].apply(pace_str_to_sec)
                        temp_df['DisplayValue'] = plot_df_base[col].astype(str)
                    else:
                        temp_df['Value'] = plot_df_base[col]
                        temp_df['DisplayValue'] = plot_df_base[col].astype(str)

                    dfs_to_concat.append(temp_df)

                chart_df = pd.concat(dfs_to_concat, ignore_index=True)
                chart_df = chart_df.dropna(subset=['Value'])

                # --- グラフ描画（Altair 共有ツールチップの確実な実装） ---

                # マウスのX位置を取得するセレクション
                hover = alt.selection_point(
                    fields=['Time'],
                    nearest=True,
                    on='mouseover',
                    empty=False,
                )

                # ベースの折れ線グラフ
                lines = alt.Chart(chart_df).mark_line().encode(
                    x=alt.X(f'Time:{x_type}', title='Time'),
                    y=alt.Y('Value:Q', title='Value', scale=alt.Scale(zero=False)),
                    color=alt.Color('Metric:N', title='Data Type')
                )

                # 透明なポイント（これがhoverを検知する）
                selectors = alt.Chart(chart_df).mark_point().encode(
                    x=f'Time:{x_type}',
                    opacity=alt.value(0),
                ).add_params(
                    hover
                )

                # ホバー位置に表示するテキスト (DisplayValue)
                text = lines.mark_text(align='left', dx=5, dy=-5).encode(
                    text=alt.condition(hover, 'DisplayValue:N', alt.value(' '))
                )

                # ホバー位置の縦線
                rules = alt.Chart(chart_df).mark_rule(color='gray').encode(
                    x=f'Time:{x_type}',
                ).transform_filter(
                    hover
                )

                # レイヤーを重ねる (selectorsを一番上にして検知しやすくする)
                interactive_chart = alt.layer(
                    lines, rules, text, selectors
                ).properties(
                    height=400
                ).interactive()

                st.altair_chart(interactive_chart, use_container_width=True)

            else:
                st.info("グラフにするデータを選択してください。")

            st.subheader("CSVダウンロード")
            csv_data = df.to_csv().encode('utf-8')
            st.download_button(
                label="解析データをCSVとしてダウンロード",
                data=csv_data,
                file_name=f"{uploaded_file.name.split('.')[0]}_data.csv",
                mime='text/csv',
            )
        else:
            st.warning("このFITファイルからは時系列データ（record）が見つかりませんでした。")

if __name__ == "__main__":
    main()