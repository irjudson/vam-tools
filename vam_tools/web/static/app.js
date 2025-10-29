const { createApp } = Vue;
const { createRouter, createWebHashHistory } = VueRouter;

// Shared utility functions
const formatBytes = (bytes) => {
    if (!bytes) return '0 B';
    const k = 1024;
    const sizes = ['B', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return Math.round(bytes / Math.pow(k, i) * 100) / 100 + ' ' + sizes[i];
};

const formatDate = (dateStr) => {
    if (!dateStr) return 'No date';
    const date = new Date(dateStr);
    return date.toLocaleDateString() + ' ' + date.toLocaleTimeString();
};

const getFileName = (path) => {
    return path.split('/').pop();
};

// Overview Component
const OverviewView = {
    template: `
        <div>
            <div style="margin-bottom: 2rem; padding: 1.5rem; background: #1e293b; border: 1px solid #334155; border-radius: 0.5rem;">
                <h3 style="font-size: 1.25rem; color: #60a5fa; margin-bottom: 1rem;">Quick Actions</h3>
                <div style="display: flex; gap: 1rem; flex-wrap: wrap;">
                    <router-link to="/files" style="padding: 0.75rem 1.5rem; background: #60a5fa; color: white; border: none; border-radius: 0.375rem; font-weight: 600; cursor: pointer; text-decoration: none; display: inline-block;">
                        📁 Browse All Files →
                    </router-link>
                    <router-link to="/duplicates" style="padding: 0.75rem 1.5rem; background: #8b5cf6; color: white; border: none; border-radius: 0.375rem; font-weight: 600; cursor: pointer; text-decoration: none; display: inline-block;">
                        🔍 Find Duplicates →
                    </router-link>
                    <router-link to="/review" style="padding: 0.75rem 1.5rem; background: #f59e0b; color: white; border: none; border-radius: 0.375rem; font-weight: 600; cursor: pointer; text-decoration: none; display: inline-block;">
                        ⚠️ Review Issues →
                    </router-link>
                </div>
            </div>

            <div v-if="dashboardStats" class="stats-grid">
                <!-- Overview Card -->
                <div class="stat-card">
                    <h3>Overview</h3>
                    <div class="value">{{ totalFiles.toLocaleString() }}</div>
                    <div class="subvalue">Total files</div>
                    <div class="value" style="font-size: 1.5rem; margin-top: 0.5rem;">{{ formatBytes(catalogInfo.statistics?.total_size_bytes || 0) }}</div>
                    <div class="subvalue">Total size</div>
                </div>

                <!-- Duplicates Card with breakdown -->
                <div class="stat-card">
                    <h3>Duplicates</h3>
                    <div class="value">{{ dashboardStats.duplicates.total_groups.toLocaleString() }}</div>
                    <div class="subvalue">Duplicate groups</div>
                    <div style="margin-top: 1rem; padding-top: 1rem; border-top: 1px solid #334155;">
                        <div style="font-size: 0.875rem; color: #94a3b8; margin-bottom: 0.25rem;">
                            {{ dashboardStats.duplicates.total_duplicate_images.toLocaleString() }} total duplicate images
                        </div>
                        <div style="font-size: 0.875rem; color: #10b981;">
                            {{ dashboardStats.duplicates.potential_space_savings_gb.toFixed(1) }} GB potential savings
                        </div>
                    </div>
                </div>

                <!-- Review Queue Card with breakdown -->
                <div class="stat-card">
                    <h3>Review Queue</h3>
                    <div class="value">{{ dashboardStats.review.total_needing_review.toLocaleString() }}</div>
                    <div class="subvalue">Files needing review</div>
                    <div style="margin-top: 1rem; padding-top: 1rem; border-top: 1px solid #334155;">
                        <div style="font-size: 0.875rem; color: #94a3b8; margin-bottom: 0.25rem;">
                            {{ dashboardStats.review.date_conflicts.toLocaleString() }} date conflicts
                        </div>
                        <div style="font-size: 0.875rem; color: #94a3b8; margin-bottom: 0.25rem;">
                            {{ dashboardStats.review.no_dates.toLocaleString() }} no dates
                        </div>
                        <div style="font-size: 0.875rem; color: #94a3b8;">
                            {{ dashboardStats.review.suspicious_dates.toLocaleString() }} suspicious dates
                        </div>
                    </div>
                </div>

                <!-- Hash Coverage Card with breakdown -->
                <div class="stat-card">
                    <h3>Hash Coverage</h3>
                    <div class="value">{{ dashboardStats.hashes.coverage_percent }}%</div>
                    <div class="subvalue">Files with perceptual hashes</div>
                    <div style="margin-top: 1rem; padding-top: 1rem; border-top: 1px solid #334155;">
                        <div style="font-size: 0.875rem; color: #94a3b8; margin-bottom: 0.25rem;">
                            {{ dashboardStats.hashes.total_hashed.toLocaleString() }} hashed
                        </div>
                        <div style="font-size: 0.875rem; color: #ef4444;">
                            {{ dashboardStats.hashes.failed_hashes.toLocaleString() }} failed
                        </div>
                    </div>
                </div>

                <!-- File Types Card -->
                <div class="stat-card">
                    <h3>File Types</h3>
                    <div style="margin-top: 0.5rem;">
                        <div style="font-size: 0.875rem; color: #94a3b8; margin-bottom: 0.5rem;">
                            <strong style="color: #60a5fa;">{{ catalogInfo.statistics?.total_images.toLocaleString() || 0 }}</strong> Images
                        </div>
                        <div style="font-size: 0.875rem; color: #94a3b8;">
                            <strong style="color: #60a5fa;">{{ catalogInfo.statistics?.total_videos.toLocaleString() || 0 }}</strong> Videos
                        </div>
                    </div>
                </div>

                <!-- Storage Card -->
                <div class="stat-card">
                    <h3>Storage</h3>
                    <div class="value">{{ formatBytes(catalogInfo.statistics?.total_size_bytes || 0) }}</div>
                    <div class="subvalue">Used space</div>
                    <div style="margin-top: 1rem; padding-top: 1rem; border-top: 1px solid #334155;">
                        <div style="font-size: 0.875rem; color: #10b981;">
                            {{ dashboardStats.duplicates.potential_space_savings_gb.toFixed(1) }} GB reclaimable
                        </div>
                    </div>
                </div>
            </div>
        </div>
    `,
    props: ['catalogInfo', 'dashboardStats'],
    computed: {
        totalFiles() {
            return (this.catalogInfo.statistics?.total_images || 0) + (this.catalogInfo.statistics?.total_videos || 0);
        }
    },
    methods: {
        formatBytes
    }
};

// All Files Component with Enhanced Grid
const AllFilesView = {
    template: `
        <div>
            <div class="controls">
                <select v-model="filterType" @change="loadImages">
                    <option value="">All Images</option>
                    <option value="image">Images Only</option>
                    <option value="video">Videos Only</option>
                    <option value="no_date">No Date</option>
                    <option value="suspicious">Suspicious Dates</option>
                </select>

                <select v-model="sortBy" @change="loadImages">
                    <option value="date">Sort by Date</option>
                    <option value="path">Sort by Path</option>
                    <option value="size">Sort by Size</option>
                </select>

                <select v-model="gridDensity">
                    <option value="compact">🔲 Compact</option>
                    <option value="comfortable">▢ Comfortable</option>
                    <option value="spacious">□ Spacious</option>
                </select>

                <select v-model.number="pageSize" @change="loadImages">
                    <option :value="20">Show 20</option>
                    <option :value="50">Show 50</option>
                    <option :value="100">Show 100</option>
                    <option :value="200">Show 200</option>
                </select>

                <button @click="toggleScrollMode" :style="scrollMode ? '' : 'background: #8b5cf6;'">
                    {{ scrollMode ? '📜 Scroll Mode' : '📄 Page Mode' }}
                </button>

                <button @click="refresh">🔄 Refresh</button>
            </div>

            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 1rem; padding-bottom: 1rem; border-bottom: 1px solid #334155;">
                <h2>
                    <span v-if="scrollMode">
                        Loaded {{ loadedCount }} images
                        <span v-if="isLoadingMore" style="color: #60a5fa; font-size: 0.875rem;">(loading more...)</span>
                        <span v-else-if="!hasMore" style="color: #94a3b8; font-size: 0.875rem;">(all loaded)</span>
                    </span>
                    <span v-else>
                        Page {{ currentPage + 1 }} ({{ images.length }} images)
                    </span>
                </h2>
            </div>

            <div class="image-grid" :class="'grid-' + gridDensity">
                <div v-for="(image, index) in images" :key="image.id" class="image-card" @click="openLightbox(index)">
                    <div class="image-thumbnail" :class="'thumbnail-' + gridDensity">
                        <img :src="'/api/images/' + image.id + '/thumbnail'"
                             :alt="image.source_path"
                             loading="lazy">
                    </div>
                    <div class="image-info">
                        <div class="image-path" :title="image.source_path">
                            {{ getFileName(image.source_path) }}
                        </div>
                        <div class="image-meta">
                            <span>{{ image.format }}</span>
                            <span v-if="image.resolution">{{ image.resolution[0] }}×{{ image.resolution[1] }}</span>
                            <span>{{ formatBytes(image.size_bytes) }}</span>
                        </div>
                        <div>
                            <span class="badge badge-info" v-if="image.selected_date">
                                {{ formatDateShort(image.selected_date) }}
                            </span>
                            <span class="badge badge-error" v-if="!image.selected_date">
                                No Date
                            </span>
                            <span class="badge badge-warning" v-if="image.suspicious">
                                Suspicious
                            </span>
                        </div>
                    </div>
                </div>
            </div>

            <div v-if="scrollMode && hasMore" style="text-align: center; padding: 2rem;">
                <button @click="loadMoreImages" :disabled="isLoadingMore" style="padding: 1rem 2rem; font-size: 1rem;">
                    {{ isLoadingMore ? 'Loading...' : 'Load More Images' }}
                </button>
            </div>
            <div v-if="scrollMode && !hasMore && loadedCount > 0" style="text-align: center; padding: 2rem; color: #64748b;">
                All images loaded ({{ loadedCount }} total)
            </div>

            <div v-if="!scrollMode" class="pagination">
                <button @click="prevPage" :disabled="currentPage === 0">
                    ← Previous
                </button>
                <span style="padding: 0 1rem;">
                    Page {{ currentPage + 1 }}
                    <span v-if="totalCount > 0" style="color: #64748b;">
                        of ~{{ Math.ceil(totalCount / pageSize) }}
                    </span>
                </span>
                <button @click="nextPage" :disabled="images.length < pageSize">
                    Next →
                </button>
            </div>

            <!-- Enhanced Lightbox -->
            <div v-if="lightboxIndex !== null" class="modal" @click="closeLightbox" @keydown.esc="closeLightbox" @keydown.left="prevImage" @keydown.right="nextImage">
                <div class="modal-content lightbox-content" @click.stop>
                    <span class="close-button" @click="closeLightbox">&times;</span>

                    <div class="lightbox-nav">
                        <button @click.stop="prevImage" :disabled="lightboxIndex === 0" class="nav-button">
                            ‹
                        </button>
                        <button @click.stop="nextImage" :disabled="lightboxIndex === images.length - 1" class="nav-button">
                            ›
                        </button>
                    </div>

                    <div class="lightbox-image-container">
                        <img :src="'/api/images/' + currentLightboxImage.id + '/file'"
                             :alt="currentLightboxImage.source_path"
                             class="lightbox-image">
                    </div>

                    <div class="lightbox-info">
                        <h3>{{ getFileName(currentLightboxImage.source_path) }}</h3>
                        <div class="detail-grid">
                            <div class="detail-item">
                                <strong>Path:</strong><br>
                                {{ currentLightboxImage.source_path }}
                            </div>
                            <div class="detail-item">
                                <strong>Format:</strong> {{ currentLightboxImage.format }}
                            </div>
                            <div class="detail-item">
                                <strong>Resolution:</strong>
                                <span v-if="currentLightboxImage.resolution">
                                    {{ currentLightboxImage.resolution[0] }} × {{ currentLightboxImage.resolution[1] }}
                                </span>
                            </div>
                            <div class="detail-item">
                                <strong>Size:</strong> {{ formatBytes(currentLightboxImage.size_bytes) }}
                            </div>
                            <div class="detail-item">
                                <strong>Date:</strong> {{ currentLightboxImage.selected_date ? formatDate(currentLightboxImage.selected_date) : 'None' }}
                            </div>
                        </div>
                        <div style="margin-top: 1rem; color: #64748b; font-size: 0.875rem;">
                            {{ lightboxIndex + 1 }} of {{ images.length }} • Use arrow keys to navigate
                        </div>
                    </div>
                </div>
            </div>
        </div>
    `,
    props: ['catalogInfo'],
    data() {
        return {
            images: [],
            lightboxIndex: null,
            filterType: '',
            sortBy: 'date',
            currentPage: 0,
            pageSize: 50,
            scrollMode: true,
            gridDensity: 'comfortable',
            loadedCount: 0,
            totalCount: 0,
            isLoadingMore: false,
            hasMore: true,
        };
    },
    computed: {
        currentLightboxImage() {
            return this.images[this.lightboxIndex];
        }
    },
    methods: {
        formatBytes,
        formatDate,
        getFileName,
        formatDateShort(dateStr) {
            if (!dateStr) return 'No date';
            const date = new Date(dateStr);
            return date.toLocaleDateString();
        },
        async loadImages() {
            try {
                const params = {
                    skip: this.scrollMode ? 0 : this.currentPage * this.pageSize,
                    limit: this.pageSize,
                    sort_by: this.sortBy,
                };
                if (this.filterType) {
                    params.filter_type = this.filterType;
                }
                const response = await axios.get('/api/images', { params });

                if (this.scrollMode) {
                    this.images = response.data;
                    this.loadedCount = response.data.length;
                    this.hasMore = response.data.length >= this.pageSize;
                    this.currentPage = 0;
                } else {
                    this.images = response.data;
                    this.loadedCount = (this.currentPage * this.pageSize) + response.data.length;
                }
            } catch (error) {
                console.error('Error loading images:', error);
            }
        },
        async loadMoreImages() {
            if (!this.scrollMode || this.isLoadingMore || !this.hasMore) {
                return;
            }

            this.isLoadingMore = true;
            try {
                const nextPage = Math.floor(this.loadedCount / this.pageSize);
                const params = {
                    skip: nextPage * this.pageSize,
                    limit: this.pageSize,
                    sort_by: this.sortBy,
                };
                if (this.filterType) {
                    params.filter_type = this.filterType;
                }
                const response = await axios.get('/api/images', { params });

                if (response.data.length > 0) {
                    this.images.push(...response.data);
                    this.loadedCount += response.data.length;
                    this.hasMore = response.data.length >= this.pageSize;
                } else {
                    this.hasMore = false;
                }
            } catch (error) {
                console.error('Error loading more images:', error);
            } finally {
                this.isLoadingMore = false;
            }
        },
        toggleScrollMode() {
            this.scrollMode = !this.scrollMode;
            this.currentPage = 0;
            this.loadImages();
        },
        openLightbox(index) {
            this.lightboxIndex = index;
            document.addEventListener('keydown', this.handleKeydown);
        },
        closeLightbox() {
            this.lightboxIndex = null;
            document.removeEventListener('keydown', this.handleKeydown);
        },
        prevImage() {
            if (this.lightboxIndex > 0) {
                this.lightboxIndex--;
            }
        },
        nextImage() {
            if (this.lightboxIndex < this.images.length - 1) {
                this.lightboxIndex++;
            }
        },
        handleKeydown(e) {
            if (e.key === 'Escape') this.closeLightbox();
            if (e.key === 'ArrowLeft') this.prevImage();
            if (e.key === 'ArrowRight') this.nextImage();
        },
        async refresh() {
            this.$emit('refresh-data');
            await this.loadImages();
        },
        async nextPage() {
            this.currentPage++;
            await this.loadImages();
        },
        async prevPage() {
            if (this.currentPage > 0) {
                this.currentPage--;
                await this.loadImages();
            }
        },
    },
    async mounted() {
        await this.loadImages();
    },
    beforeUnmount() {
        document.removeEventListener('keydown', this.handleKeydown);
    },
};

// Duplicates Component (unchanged)
const DuplicatesView = {
    template: `
        <div>
            <div v-if="loading" class="loading">
                <div class="spinner"></div>
                <p>Loading duplicate groups...</p>
            </div>

            <div v-else>
                <div class="stats-grid">
                    <div class="stat-card">
                        <h3>Duplicate Groups</h3>
                        <div class="value">{{ stats.total_groups || 0 }}</div>
                        <div class="subvalue">{{ stats.total_duplicates || 0 }} total images</div>
                    </div>
                    <div class="stat-card">
                        <h3>Needs Review</h3>
                        <div class="value">{{ stats.needs_review || 0 }}</div>
                        <div class="subvalue">Groups requiring attention</div>
                    </div>
                    <div class="stat-card">
                        <h3>Space Savings</h3>
                        <div class="value">{{ formatBytes(stats.potential_space_savings_bytes || 0) }}</div>
                        <div class="subvalue">If duplicates removed</div>
                    </div>
                </div>

                <div class="controls">
                    <select v-model="filterReview" @change="loadGroups">
                        <option value="all">All Groups</option>
                        <option value="review">Needs Review Only</option>
                    </select>
                    <button @click="loadGroups">🔄 Refresh</button>
                </div>

                <div v-if="groups.length === 0" style="text-align: center; padding: 3rem; color: #94a3b8;">
                    <p style="font-size: 1.5rem; margin-bottom: 0.5rem;">✨ No duplicates found</p>
                    <p>Run duplicate detection analysis to find similar images</p>
                </div>

                <div v-for="group in groups" :key="group.id" class="duplicate-group">
                    <div class="duplicate-header">
                        <div>
                            <h3 style="color: #e2e8f0; margin-bottom: 0.25rem;">
                                Group {{ group.id.substring(0, 8) }}
                            </h3>
                            <span style="font-size: 0.875rem; color: #94a3b8;">
                                {{ group.duplicate_count }} images •
                                {{ formatBytes(group.total_size_bytes) }} total •
                                Formats: {{ group.format_types.join(', ') || 'Unknown' }}
                            </span>
                        </div>
                        <div>
                            <span v-if="group.needs_review" class="duplicate-badge">NEEDS REVIEW</span>
                            <button @click="compareGroup(group)" style="margin-left: 0.5rem;">
                                👁️ Compare
                            </button>
                        </div>
                    </div>

                    <div class="comparison-grid" v-if="groupImages[group.id]">
                        <div v-for="imageId in groupImages[group.id]" :key="imageId"
                             :class="['image-comparison', imageId === group.primary_image_id ? 'primary' : '']">
                            <div class="image-comparison-header">
                                <span style="font-size: 0.75rem; color: #94a3b8;">
                                    {{ getImageFileName(imageId) }}
                                </span>
                                <span v-if="imageId === group.primary_image_id" class="primary-badge">
                                    ⭐ PRIMARY
                                </span>
                            </div>
                            <div class="image-preview" @click="viewFullImage(imageId, group.id)">
                                <img :src="'/api/images/' + imageId + '/file'" :alt="imageId" loading="lazy">
                            </div>
                            <div class="image-details" v-if="imageDetails[imageId]">
                                <div class="detail-row">
                                    <span>Format:</span>
                                    <strong>{{ imageDetails[imageId].format || 'N/A' }}</strong>
                                </div>
                                <div class="detail-row">
                                    <span>Resolution:</span>
                                    <strong v-if="imageDetails[imageId].resolution">
                                        {{ imageDetails[imageId].resolution[0] }}×{{ imageDetails[imageId].resolution[1] }}
                                    </strong>
                                    <strong v-else>N/A</strong>
                                </div>
                                <div class="detail-row">
                                    <span>Size:</span>
                                    <strong>{{ formatBytes(imageDetails[imageId].size_bytes || 0) }}</strong>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>

            <div v-if="comparisonModal" class="modal" @click="comparisonModal = null">
                <div class="modal-content" @click.stop style="max-width: 90vw;">
                    <span class="close-button" @click="comparisonModal = null">&times;</span>
                    <h2>Side-by-Side Comparison</h2>
                    <p style="color: #94a3b8; margin-bottom: 1rem;">
                        Group {{ comparisonModal.group.id.substring(0, 8) }}
                    </p>

                    <div class="comparison-modal">
                        <div v-for="imageId in comparisonModal.images" :key="imageId" class="comparison-image">
                            <div style="margin-bottom: 0.5rem;">
                                <span v-if="imageId === comparisonModal.group.primary_image_id"
                                      class="primary-badge">⭐ PRIMARY</span>
                                <span v-else class="duplicate-badge">DUPLICATE</span>
                            </div>
                            <img :src="'/api/images/' + imageId + '/file'" :alt="imageId">
                            <div v-if="imageDetails[imageId]" style="text-align: left; margin-top: 1rem;">
                                <div class="detail-row">
                                    <span>Path:</span>
                                    <strong style="font-size: 0.75rem;">{{ imageDetails[imageId].source_path }}</strong>
                                </div>
                                <div class="detail-row">
                                    <span>Format:</span>
                                    <strong>{{ imageDetails[imageId].format }}</strong>
                                </div>
                                <div class="detail-row">
                                    <span>Resolution:</span>
                                    <strong v-if="imageDetails[imageId].resolution">
                                        {{ imageDetails[imageId].resolution[0] }}×{{ imageDetails[imageId].resolution[1] }}
                                    </strong>
                                </div>
                                <div class="detail-row">
                                    <span>Size:</span>
                                    <strong>{{ formatBytes(imageDetails[imageId].size_bytes || 0) }}</strong>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    `,
    data() {
        return {
            loading: true,
            stats: {},
            groups: [],
            groupImages: {},
            imageDetails: {},
            filterReview: 'all',
            comparisonModal: null,
        };
    },
    methods: {
        formatBytes,
        async loadStats() {
            try {
                const response = await axios.get('/api/duplicates/stats');
                this.stats = response.data;
            } catch (error) {
                console.error('Error loading duplicate stats:', error);
            }
        },
        async loadGroups() {
            try {
                const params = {
                    needs_review_only: this.filterReview === 'review',
                };
                const response = await axios.get('/api/duplicates/groups', { params });
                this.groups = response.data;

                for (const group of this.groups) {
                    await this.loadGroupDetail(group.id);
                }
            } catch (error) {
                console.error('Error loading duplicate groups:', error);
            }
        },
        async loadGroupDetail(groupId) {
            try {
                const response = await axios.get(`/api/duplicates/groups/${groupId}`);
                this.groupImages[groupId] = response.data.duplicate_image_ids;

                for (const imageId of response.data.duplicate_image_ids) {
                    if (!this.imageDetails[imageId]) {
                        await this.loadImageDetail(imageId);
                    }
                }
            } catch (error) {
                console.error('Error loading group detail:', error);
            }
        },
        async loadImageDetail(imageId) {
            try {
                const response = await axios.get(`/api/images/${imageId}`);
                this.imageDetails[imageId] = response.data;
            } catch (error) {
                console.error('Error loading image detail:', error);
            }
        },
        compareGroup(group) {
            this.comparisonModal = {
                group: group,
                images: this.groupImages[group.id] || [],
            };
        },
        viewFullImage(imageId, groupId) {
            this.comparisonModal = {
                group: this.groups.find(g => g.id === groupId),
                images: this.groupImages[groupId] || [],
            };
        },
        getImageFileName(imageId) {
            const detail = this.imageDetails[imageId];
            if (detail && detail.source_path) {
                return detail.source_path.split('/').pop();
            }
            return imageId.substring(0, 8);
        },
    },
    async mounted() {
        await this.loadStats();
        await this.loadGroups();
        this.loading = false;
    },
};

// Review Component (unchanged from before)
const ReviewView = {
    template: `
        <div>
            <div v-if="loading" class="loading">
                <div class="spinner"></div>
                <p>Loading review queue...</p>
            </div>

            <div v-else>
                <div class="stats-grid">
                    <div class="stat-card"
                         :class="{ active: filterType === null }"
                         @click="setFilter(null)">
                        <h3>All Issues</h3>
                        <div class="value">{{ stats.total_issues }}</div>
                    </div>
                    <div class="stat-card"
                         :class="{ active: filterType === 'date_conflict' }"
                         @click="setFilter('date_conflict')">
                        <h3>Date Conflicts</h3>
                        <div class="value">{{ stats.date_conflicts }}</div>
                    </div>
                    <div class="stat-card"
                         :class="{ active: filterType === 'no_date' }"
                         @click="setFilter('no_date')">
                        <h3>No Date</h3>
                        <div class="value">{{ stats.no_date }}</div>
                    </div>
                    <div class="stat-card"
                         :class="{ active: filterType === 'suspicious_date' }"
                         @click="setFilter('suspicious_date')">
                        <h3>Suspicious Dates</h3>
                        <div class="value">{{ stats.suspicious_dates }}</div>
                    </div>
                    <div class="stat-card"
                         :class="{ active: filterType === 'low_confidence' }"
                         @click="setFilter('low_confidence')">
                        <h3>Low Confidence</h3>
                        <div class="value">{{ stats.low_confidence }}</div>
                    </div>
                </div>

                <div v-if="selectedItems.length > 0" class="batch-actions">
                    <span class="batch-actions-label">{{ selectedItems.length }} selected:</span>
                    <button @click="showBatchDatePicker = !showBatchDatePicker" class="secondary">
                        📅 Set Date for All
                    </button>
                    <button @click="clearSelection" class="secondary">
                        Clear Selection
                    </button>
                </div>

                <div v-if="showBatchDatePicker && selectedItems.length > 0" class="batch-actions">
                    <input type="date" v-model="batchDate">
                    <input type="time" v-model="batchTime">
                    <button @click="applyBatchDate" class="success">
                        ✓ Apply to {{ selectedItems.length }} items
                    </button>
                    <button @click="showBatchDatePicker = false" class="secondary">
                        Cancel
                    </button>
                </div>

                <div class="controls">
                    <button @click="refresh">🔄 Refresh</button>
                    <button @click="selectAll" class="secondary">
                        {{ selectedItems.length === items.length && items.length > 0 ? 'Deselect All' : 'Select All' }}
                    </button>
                    <span style="color: #64748b; margin-left: auto;">
                        Showing {{ items.length }} items
                    </span>
                </div>

                <div v-if="items.length === 0" class="empty-state">
                    <h3>✓ All Clear!</h3>
                    <p>No items need review{{ filterType ? ' in this category' : '' }}.</p>
                </div>

                <div v-else class="review-grid">
                    <div v-for="item in items"
                         :key="item.id"
                         class="review-card"
                         :class="{ selected: isSelected(item.id) }">
                        <div class="review-image" @click="showImageDetail(item)">
                            <img :src="'/api/images/' + item.id + '/file'"
                                 :alt="item.source_path"
                                 loading="lazy">
                        </div>
                        <div class="review-info">
                            <div class="checkbox-label">
                                <input type="checkbox"
                                       :checked="isSelected(item.id)"
                                       @change="toggleSelection(item.id)">
                                <span class="review-path" :title="item.source_path">
                                    {{ getFileName(item.source_path) }}
                                </span>
                            </div>

                            <div style="margin-bottom: 0.75rem;">
                                <span v-if="item.issue_type === 'date_conflict'" class="badge badge-warning">
                                    Date Conflict
                                </span>
                                <span v-if="item.issue_type === 'no_date'" class="badge badge-error">
                                    No Date
                                </span>
                                <span v-if="item.issue_type === 'suspicious_date'" class="badge badge-warning">
                                    Suspicious
                                </span>
                                <span v-if="item.issue_type === 'low_confidence'" class="badge badge-info">
                                    Low Confidence ({{ item.confidence }}%)
                                </span>
                            </div>

                            <div class="review-dates">
                                <div v-if="item.dates" style="font-size: 0.75rem; color: #64748b; margin-bottom: 0.5rem;">
                                    Available dates:
                                </div>
                                <div v-if="item.dates?.exif_date" class="date-row">
                                    <span class="date-label">EXIF:</span>
                                    <span class="date-value">{{ formatDate(item.dates.exif_date) }}</span>
                                </div>
                                <div v-if="item.dates?.filename_date" class="date-row">
                                    <span class="date-label">Filename:</span>
                                    <span class="date-value">{{ formatDate(item.dates.filename_date) }}</span>
                                </div>
                                <div v-if="item.dates?.directory_date" class="date-row">
                                    <span class="date-label">Directory:</span>
                                    <span class="date-value">{{ formatDate(item.dates.directory_date) }}</span>
                                </div>
                                <div v-if="item.dates?.filesystem_date" class="date-row">
                                    <span class="date-label">File System:</span>
                                    <span class="date-value">{{ formatDate(item.dates.filesystem_date) }}</span>
                                </div>
                                <div v-if="!item.dates || Object.keys(item.dates).length === 0" class="date-row">
                                    <span class="date-value none">No dates available</span>
                                </div>
                            </div>

                            <div class="date-picker-section">
                                <h4>Set Manual Date</h4>
                                <div class="date-inputs">
                                    <input type="date"
                                           :value="getItemDate(item.id)"
                                           @input="updateItemDate(item.id, $event.target.value)">
                                    <input type="time"
                                           :value="getItemTime(item.id)"
                                           @input="updateItemTime(item.id, $event.target.value)">
                                </div>
                            </div>

                            <div class="action-buttons">
                                <button @click="saveDate(item)"
                                        class="success"
                                        :disabled="!hasManualDate(item.id)">
                                    ✓ Save Date
                                </button>
                                <button @click="skipItem(item.id)" class="secondary">
                                    Skip
                                </button>
                            </div>
                        </div>
                    </div>
                </div>
            </div>

            <div v-if="detailImage" class="modal" @click="detailImage = null">
                <div class="modal-content" @click.stop>
                    <span class="close-button" @click="detailImage = null">&times;</span>
                    <h2>{{ getFileName(detailImage.source_path) }}</h2>
                    <img :src="'/api/images/' + detailImage.id + '/file'"
                         class="modal-image"
                         :alt="detailImage.source_path">
                    <div style="font-size: 0.875rem; color: #94a3b8; margin-top: 1rem;">
                        <strong>Full path:</strong><br>
                        {{ detailImage.source_path }}
                    </div>
                </div>
            </div>
        </div>
    `,
    data() {
        return {
            loading: true,
            stats: {
                total_issues: 0,
                date_conflicts: 0,
                no_date: 0,
                suspicious_dates: 0,
                low_confidence: 0,
            },
            items: [],
            filterType: null,
            selectedItems: [],
            manualDates: {},
            showBatchDatePicker: false,
            batchDate: '',
            batchTime: '',
            detailImage: null,
        };
    },
    methods: {
        formatDate,
        getFileName,
        async loadStats() {
            try {
                const response = await axios.get('/api/review/stats');
                this.stats = response.data;
            } catch (error) {
                console.error('Error loading stats:', error);
            }
        },
        async loadItems() {
            try {
                const params = {};
                if (this.filterType) {
                    params.filter_type = this.filterType;
                }
                const response = await axios.get('/api/review/queue', { params });
                this.items = response.data;
            } catch (error) {
                console.error('Error loading items:', error);
            }
        },
        async refresh() {
            await this.loadStats();
            await this.loadItems();
            this.selectedItems = [];
            this.manualDates = {};
            this.showBatchDatePicker = false;
        },
        setFilter(filterType) {
            this.filterType = filterType;
            this.loadItems();
            this.selectedItems = [];
        },
        isSelected(imageId) {
            return this.selectedItems.includes(imageId);
        },
        toggleSelection(imageId) {
            const index = this.selectedItems.indexOf(imageId);
            if (index > -1) {
                this.selectedItems.splice(index, 1);
            } else {
                this.selectedItems.push(imageId);
            }
        },
        selectAll() {
            if (this.selectedItems.length === this.items.length) {
                this.selectedItems = [];
            } else {
                this.selectedItems = this.items.map(item => item.id);
            }
        },
        clearSelection() {
            this.selectedItems = [];
            this.showBatchDatePicker = false;
        },
        getItemDate(imageId) {
            return this.manualDates[imageId]?.date || '';
        },
        getItemTime(imageId) {
            return this.manualDates[imageId]?.time || '12:00';
        },
        updateItemDate(imageId, date) {
            if (!this.manualDates[imageId]) {
                this.manualDates[imageId] = {};
            }
            this.manualDates[imageId].date = date;
        },
        updateItemTime(imageId, time) {
            if (!this.manualDates[imageId]) {
                this.manualDates[imageId] = {};
            }
            this.manualDates[imageId].time = time;
        },
        hasManualDate(imageId) {
            return this.manualDates[imageId]?.date;
        },
        async saveDate(item) {
            const dateData = this.manualDates[item.id];
            if (!dateData?.date) {
                alert('Please select a date first');
                return;
            }

            try {
                const dateStr = `${dateData.date}T${dateData.time || '12:00'}:00`;
                await axios.patch(`/api/images/${item.id}/date?date_str=${encodeURIComponent(dateStr)}`);

                this.items = this.items.filter(i => i.id !== item.id);
                delete this.manualDates[item.id];

                await this.loadStats();
            } catch (error) {
                console.error('Error saving date:', error);
                alert('Failed to save date: ' + (error.response?.data?.detail || error.message));
            }
        },
        async applyBatchDate() {
            if (!this.batchDate) {
                alert('Please select a date');
                return;
            }

            const dateStr = `${this.batchDate}T${this.batchTime || '12:00'}:00`;
            const selectedIds = [...this.selectedItems];

            let successCount = 0;
            let errorCount = 0;

            for (const imageId of selectedIds) {
                try {
                    await axios.patch(`/api/images/${imageId}/date?date_str=${encodeURIComponent(dateStr)}`);
                    successCount++;

                    this.items = this.items.filter(i => i.id !== imageId);
                } catch (error) {
                    console.error(`Error saving date for ${imageId}:`, error);
                    errorCount++;
                }
            }

            alert(`Updated ${successCount} items${errorCount > 0 ? `, ${errorCount} failed` : ''}`);

            this.selectedItems = [];
            this.showBatchDatePicker = false;
            this.batchDate = '';
            this.batchTime = '';

            await this.loadStats();
        },
        skipItem(imageId) {
            this.items = this.items.filter(i => i.id !== imageId);
            delete this.manualDates[imageId];
            const index = this.selectedItems.indexOf(imageId);
            if (index > -1) {
                this.selectedItems.splice(index, 1);
            }
        },
        showImageDetail(item) {
            this.detailImage = item;
        },
    },
    async mounted() {
        await this.loadStats();
        await this.loadItems();
        this.loading = false;

        this.batchTime = '12:00';
    },
};

// Router configuration
const router = createRouter({
    history: createWebHashHistory(),
    routes: [
        { path: '/', component: OverviewView, props: true },
        { path: '/files', component: AllFilesView, props: true },
        { path: '/duplicates', component: DuplicatesView },
        { path: '/review', component: ReviewView }
    ]
});

// Main app
createApp({
    data() {
        return {
            globalLoading: true,
            catalogInfo: {},
            dashboardStats: null,
        };
    },
    methods: {
        async loadCatalogInfo() {
            try {
                const response = await axios.get('/api/catalog/info');
                this.catalogInfo = response.data;
            } catch (error) {
                console.error('Error loading catalog info:', error);
            }
        },
        async loadDashboardStats() {
            try {
                const response = await axios.get('/api/dashboard/stats');
                this.dashboardStats = response.data;
            } catch (error) {
                console.error('Error loading dashboard stats:', error);
            }
        },
        async loadGlobalData() {
            await this.loadCatalogInfo();
            await this.loadDashboardStats();
        },
    },
    async mounted() {
        await this.loadGlobalData();
        this.globalLoading = false;
    },
}).use(router).mount('#app');
