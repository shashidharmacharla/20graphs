import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt

# Metallic color palette
metallic_colors = [
    "#D4AF37",  # Gold
    "#C0C0C0",  # Silver
    "#B87333",  # Copper/Bronze
    "#A9A9A9",  # Dark Gray / Steel
    "#8A795D",  # Antique Bronze
    "#B0B7BC",  # Light Steel Blue
    "#7C7C7C",  # Gray
    "#FFD700",  # Bright Gold
]

st.set_page_config(page_title="Metalic Analytics Dashboard", layout="wide")

st.title("📊 Metallic Colors Analytics Dashboard")

# Upload dataset
uploaded_file = st.file_uploader("Upload your CSV dataset", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success(f"Dataset loaded successfully! Shape: {df.shape}")

    # Show raw data option
    if st.checkbox("Show raw data"):
        st.dataframe(df)

    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

    if len(numeric_cols) == 0:
        st.warning("Dataset has no numeric columns for analysis.")
    else:
        st.markdown("### Select Column(s) for Analysis")
        num_col = st.selectbox("Numeric column", numeric_cols)
        cat_col = None
        if len(categorical_cols) > 0:
            cat_col = st.selectbox("Categorical column (optional)", [None] + categorical_cols)

        st.markdown("---")

        # Prepare tabs for better layout
        tabs = st.tabs([
            "Distribution",
            "Trends & Time",
            "Relationships",
            "Categorical Analysis",
            "Advanced"
        ])

        # 1. Distribution Tab
        with tabs[0]:
            st.header("Distribution Plots")

            # Histogram
            fig1 = px.histogram(df, x=num_col, nbins=30,
                                color_discrete_sequence=[metallic_colors[0]])
            st.plotly_chart(fig1, use_container_width=True)

            # Boxplot
            fig2 = px.box(df, y=num_col, color_discrete_sequence=[metallic_colors[1]])
            st.plotly_chart(fig2, use_container_width=True)

            # Violin plot
            if cat_col:
                fig3 = px.violin(df, y=num_col, x=cat_col,
                                 color=cat_col,
                                 color_discrete_sequence=metallic_colors)
                st.plotly_chart(fig3, use_container_width=True)

            # Density plot (KDE)
            plt.figure(figsize=(8,4))
            sns.kdeplot(df[num_col], shade=True, color=metallic_colors[3])
            plt.title(f'Density Plot of {num_col}')
            st.pyplot(plt.gcf())
            plt.clf()

        # 2. Trends & Time Tab
        with tabs[1]:
            st.header("Trend and Time Series")

            # If datetime column present
            datetime_cols = df.select_dtypes(include=['datetime', 'object']).columns.tolist()
            dt_col = None
            for col in datetime_cols:
                try:
                    df[col] = pd.to_datetime(df[col])
                    dt_col = col
                    break
                except:
                    continue
            if dt_col:
                st.write(f"Using datetime column: **{dt_col}**")
                df_sorted = df.sort_values(dt_col)
                fig4 = px.line(df_sorted, x=dt_col, y=num_col,
                               color_discrete_sequence=[metallic_colors[4]])
                st.plotly_chart(fig4, use_container_width=True)

                if cat_col:
                    fig5 = px.line(df_sorted, x=dt_col, y=num_col, color=cat_col,
                                   color_discrete_sequence=metallic_colors)
                    st.plotly_chart(fig5, use_container_width=True)
            else:
                st.info("No datetime column detected for time series plots.")

            # Rolling Mean
            if dt_col:
                window = st.slider("Rolling window size", 1, 30, 7)
                df_sorted['rolling_mean'] = df_sorted[num_col].rolling(window).mean()
                fig6 = px.line(df_sorted, x=dt_col, y=['rolling_mean', num_col],
                               color_discrete_sequence=[metallic_colors[0], metallic_colors[1]],
                               labels={"value":"Value", "variable":"Legend"})
                st.plotly_chart(fig6, use_container_width=True)

        # 3. Relationships Tab
        with tabs[2]:
            st.header("Relationships Between Variables")

            # Scatter plot
            if len(numeric_cols) >= 2:
                num_col2 = st.selectbox("Select second numeric column", [c for c in numeric_cols if c != num_col])
                fig7 = px.scatter(df, x=num_col, y=num_col2,
                                  color=cat_col if cat_col else None,
                                  color_discrete_sequence=metallic_colors)
                st.plotly_chart(fig7, use_container_width=True)

            # Correlation heatmap
            corr = df[numeric_cols].corr()
            plt.figure(figsize=(8,6))
            sns.heatmap(corr, annot=True, cmap="coolwarm", linewidths=0.5)
            plt.title("Correlation Heatmap")
            st.pyplot(plt.gcf())
            plt.clf()

            # Pairplot (using seaborn)
            if st.checkbox("Show pairplot (takes time)"):
                sns.pairplot(df[numeric_cols], diag_kind='kde',
                             plot_kws={'color': metallic_colors[2]})
                st.pyplot(plt.gcf())
                plt.clf()

        # 4. Categorical Analysis Tab
        with tabs[3]:
            st.header("Categorical Data Analysis")

            if cat_col:
                # Countplot
                fig8 = px.histogram(df, x=cat_col, color=cat_col,
                                   color_discrete_sequence=metallic_colors)
                st.plotly_chart(fig8, use_container_width=True)

                # Boxplot numeric vs categorical
                fig9 = px.box(df, x=cat_col, y=num_col,
                              color=cat_col, color_discrete_sequence=metallic_colors)
                st.plotly_chart(fig9, use_container_width=True)

                # Bar plot - mean value per category
                mean_vals = df.groupby(cat_col)[num_col].mean().reset_index()
                fig10 = px.bar(mean_vals, x=cat_col, y=num_col,
                               color=cat_col, color_discrete_sequence=metallic_colors)
                st.plotly_chart(fig10, use_container_width=True)

                # Pie chart of category distribution
                pie_data = df[cat_col].value_counts().reset_index()
                pie_data.columns = [cat_col, 'count']
                fig11 = px.pie(pie_data, values='count', names=cat_col,
                              color_discrete_sequence=metallic_colors)
                st.plotly_chart(fig11, use_container_width=True)
            else:
                st.info("No categorical column selected for categorical analysis.")

        # 5. Advanced Tab
        with tabs[4]:
            st.header("Advanced Visualizations")

            # 3D Scatter Plot
            if len(numeric_cols) >= 3:
                num_col3 = st.selectbox("Select third numeric column", [c for c in numeric_cols if c not in [num_col, num_col2]])
                fig12 = px.scatter_3d(df, x=num_col, y=num_col2, z=num_col3,
                                      color=cat_col if cat_col else None,
                                      color_discrete_sequence=metallic_colors)
                st.plotly_chart(fig12, use_container_width=True)

            # Sunburst Chart for categorical columns (if 2+ categorical cols)
            if len(categorical_cols) >= 2:
                cat_cols = st.multiselect("Select 2 or more categorical columns for Sunburst", categorical_cols, default=categorical_cols[:2])
                if len(cat_cols) >= 2:
                    sunburst_data = df.groupby(cat_cols).size().reset_index(name='count')
                    fig13 = px.sunburst(sunburst_data, path=cat_cols, values='count',
                                        color=cat_cols[0], color_discrete_sequence=metallic_colors)
                    st.plotly_chart(fig13, use_container_width=True)

            # Heatmap of pivot table (numeric mean by categories)
            if len(categorical_cols) >= 2:
                pivot_cat1 = categorical_cols[0]
                pivot_cat2 = categorical_cols[1]
                pivot_table = df.pivot_table(index=pivot_cat1, columns=pivot_cat2, values=num_col, aggfunc='mean')
                plt.figure(figsize=(10,6))
                sns.heatmap(pivot_table, annot=True, cmap="YlGnBu", linewidths=0.3)
                plt.title(f"Heatmap of {num_col} mean by {pivot_cat1} and {pivot_cat2}")
                st.pyplot(plt.gcf())
                plt.clf()

            # Radar chart (spider chart) for category means
            if cat_col:
                means = df.groupby(cat_col)[num_col].mean()
                categories = means.index.tolist()
                values = means.values.tolist()
                values += values[:1]  # close loop

                angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
                angles += angles[:1]

                fig14 = go.Figure(
                    data=[go.Scatterpolar(r=values, theta=categories + [categories[0]],
                                          fill='toself', line_color=metallic_colors[0])]
                )
                fig14.update_layout(
                    polar=dict(radialaxis=dict(visible=True)),
                    showlegend=False,
                    title=f"Radar Chart: Mean {num_col} by {cat_col}"
                )
                st.plotly_chart(fig14, use_container_width=True)

            # Donut chart example
            if cat_col:
                donut_data = df[cat_col].value_counts().reset_index()
                donut_data.columns = [cat_col, 'count']
                fig15 = go.Figure(data=[go.Pie(labels=donut_data[cat_col], values=donut_data['count'],
                                              hole=.5,
                                              marker_colors=metallic_colors)])
                fig15.update_layout(title="Donut Chart of Categories")
                st.plotly_chart(fig15, use_container_width=True)

            # Funnel chart example (simulate stages)
            funnel_stages = ['Visited', 'Signed Up', 'Purchased', 'Retained']
            funnel_values = [1000, 600, 300, 150]
            fig16 = go.Figure(go.Funnel(
                y = funnel_stages,
                x = funnel_values,
                marker={"color": metallic_colors[:len(funnel_stages)]}
            ))
            fig16.update_layout(title="Funnel Chart Example")
            st.plotly_chart(fig16, use_container_width=True)

            # Waterfall Chart example
            waterfall_x = ['Start', 'Sales', 'Returns', 'Revenue', 'Expenses', 'Profit']
            waterfall_y = [1000, 300, -50, 1250, -400, 850]
            fig17 = go.Figure(go.Waterfall(
                x=waterfall_x,
                y=waterfall_y,
                decreasing={"marker":{"color": metallic_colors[2]}},
                increasing={"marker":{"color": metallic_colors[0]}},
                totals={"marker":{"color": metallic_colors[1]}}
            ))
            fig17.update_layout(title="Waterfall Chart Example")
            st.plotly_chart(fig17, use_container_width=True)

            # Timeline chart (simple)
            timeline_data = pd.DataFrame({
                'Task': ['Task A', 'Task B', 'Task C'],
                'Start': pd.to_datetime(['2026-01-01', '2026-01-05', '2026-01-10']),
                'Finish': pd.to_datetime(['2026-01-07', '2026-01-14', '2026-01-15']),
            })
            fig18 = px.timeline(timeline_data, x_start="Start", x_end="Finish", y="Task",
                                color_discrete_sequence=metallic_colors)
            fig18.update_yaxes(autorange="reversed")
            fig18.update_layout(title="Timeline Chart Example")
            st.plotly_chart(fig18, use_container_width=True)

            # Hexbin plot example
            if len(numeric_cols) >= 2:
                plt.figure(figsize=(8,6))
                plt.hexbin(df[num_col], df[num_col2], gridsize=30, cmap='copper')
                plt.xlabel(num_col)
                plt.ylabel(num_col2)
                plt.title("Hexbin Plot Example")
                st.pyplot(plt.gcf())
                plt.clf()

            # Area chart
            df_area = df.sort_values(dt_col) if dt_col else df
            fig19 = px.area(df_area, x=dt_col if dt_col else df.index, y=num_col,
                            color_discrete_sequence=[metallic_colors[5]])
            st.plotly_chart(fig19, use_container_width=True)

            # Barpolar chart example
            theta = ['A', 'B', 'C', 'D', 'E', 'F']
            r = [10, 20, 30, 20, 15, 25]
            fig20 = go.Figure(go.Barpolar(
                r=r,
                theta=theta,
                marker_color=metallic_colors,
                marker_line_color="black",
                marker_line_width=1.5,
                opacity=0.8
            ))
            fig20.update_layout(title="Barpolar Chart Example",
                                template=None,
                                polar=dict(bgcolor='rgba(0,0,0,0)'))
            st.plotly_chart(fig20, use_container_width=True)


else:
    st.info("Please upload a CSV file to get started.")
