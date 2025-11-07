// VAM Tools - Unified Application with Multi-Catalog Support
const { createApp } = Vue;

createApp({
    data() {
        return {
            // Navigation
            currentView: 'dashboard',
            
            // Catalog management
            catalogs: [],
            currentCatalog: null,
            showCatalogManager: false,
            showAddCatalogForm: false,
            
            addCatalogForm: {
                name: '',
                catalog_path: '',
                source_directories: '',
                description: '',
                color: '#60a5fa'
            },
            
            // Catalog data
            catalogInfo: null,
            dashboardStats: null,
            images: [],
            loading: false,
            error: null,
            
            // Jobs data
            allJobs: [],
            jobsLoading: false,
            jobsRefreshInterval: null,
            jobDetailsCache: {},
            
            // Filters and search
            currentFilter: 'all',
            searchQuery: '',
            sortBy: 'captured_date_desc',
            
            // Quick action forms
            showAnalyzeForm: false,
            showOrganizeForm: false,
            showThumbnailsForm: false,
            
            analyzeForm: {
                catalog_id: null,
                detect_duplicates: false
            },
            organizeForm: {
                catalog_id: null,
                output_directory: '',
                dry_run: true,
                operation: 'copy',
                pattern: '{year}/{month}'
            },
            thumbnailsForm: {
                catalog_id: null,
                sizes: '200, 400',
                quality: 85,
                skip_existing: true
            },
            
            // Notifications
            notifications: []
        };
    },
    
    computed: {
        filteredImages() {
            let filtered = this.images;
            
            if (this.currentFilter !== 'all') {
                filtered = filtered.filter(img => {
                    switch(this.currentFilter) {
                        case 'duplicates': return img.is_duplicate;
                        case 'no_date': return !img.captured_date;
                        case 'videos': return img.file_type === 'video';
                        default: return true;
                    }
                });
            }
            
            if (this.searchQuery) {
                const query = this.searchQuery.toLowerCase();
                filtered = filtered.filter(img => 
                    img.file_name.toLowerCase().includes(query) ||
                    (img.tags && img.tags.some(t => t.toLowerCase().includes(query)))
                );
            }
            
            return filtered;
        },
        
        jobs() {
            return this.allJobs.map(jobId => {
                const details = this.jobDetailsCache[jobId] || {};
                return {
                    id: jobId,
                    name: this.getJobName(jobId, details),
                    status: details.status || 'PENDING',
                    progress: details.progress?.percent || 0,
                    started: new Date().toISOString(),
                    ...details
                };
            });
        },
        
        activeJobs() {
            return this.jobs.filter(job => 
                job.status === 'PENDING' || job.status === 'PROGRESS'
            );
        },
        
        hasActiveJobs() {
            return this.activeJobs.length > 0;
        }
    },
    
    methods: {
        // Navigation
        setView(view) {
            this.currentView = view;
            if (view === 'jobs') {
                this.startJobsRefresh();
            }
            if (view === 'browse' && this.images.length === 0) {
                this.loadImages();
            }
        },
        
        // Catalog Management
        async loadCatalogs() {
            try {
                const response = await axios.get('/api/catalogs');
                this.catalogs = response.data;
                
                // Load current catalog
                const currentResponse = await axios.get('/api/catalogs/current');
                this.currentCatalog = currentResponse.data;
                
                // Set form defaults to current catalog if available
                if (this.currentCatalog) {
                    this.analyzeForm.catalog_id = this.currentCatalog.id;
                    this.organizeForm.catalog_id = this.currentCatalog.id;
                    this.thumbnailsForm.catalog_id = this.currentCatalog.id;
                }
            } catch (error) {
                console.error('Error loading catalogs:', error);
            }
        },
        
        async switchCatalog(catalogId) {
            try {
                await axios.post('/api/catalogs/current', { catalog_id: catalogId });
                await this.loadCatalogs();
                this.addNotification(`Switched to catalog: ${this.currentCatalog.name}`, 'success');
                
                // Reload dashboard stats for new catalog
                this.loadDashboardStats();
            } catch (error) {
                this.addNotification('Failed to switch catalog', 'error');
                console.error(error);
            }
        },
        
        async submitAddCatalog() {
            try {
                const sourceDirs = this.addCatalogForm.source_directories
                    .split('\n')
                    .map(s => s.trim())
                    .filter(s => s.length > 0);
                
                const response = await axios.post('/api/catalogs', {
                    name: this.addCatalogForm.name,
                    catalog_path: this.addCatalogForm.catalog_path,
                    source_directories: sourceDirs,
                    description: this.addCatalogForm.description,
                    color: this.addCatalogForm.color
                });
                
                this.addNotification('Catalog added successfully', 'success');
                this.showAddCatalogForm = false;
                
                // Reset form
                this.addCatalogForm = {
                    name: '',
                    catalog_path: '',
                    source_directories: '',
                    description: '',
                    color: '#60a5fa'
                };
                
                await this.loadCatalogs();
            } catch (error) {
                this.addNotification('Failed to add catalog: ' + (error.response?.data?.detail || error.message), 'error');
                console.error(error);
            }
        },
        
        async deleteCatalog(catalogId) {
            if (!confirm('Are you sure you want to delete this catalog configuration? This will not delete any files.')) {
                return;
            }
            
            try {
                await axios.delete(`/api/catalogs/${catalogId}`);
                this.addNotification('Catalog deleted', 'info');
                await this.loadCatalogs();
            } catch (error) {
                this.addNotification('Failed to delete catalog', 'error');
                console.error(error);
            }
        },
        
        getJobName(jobId, details) {
            if (details.result && details.result.catalog_path) {
                return 'Catalog Job';
            }
            return jobId.substring(0, 8) + '...';
        },
        
        // Catalog data methods
        async loadCatalogInfo() {
            try {
                const response = await axios.get('/api/catalog/info');
                this.catalogInfo = response.data;
            } catch (error) {
                console.log('No catalog info available');
            }
        },
        
        async loadDashboardStats() {
            this.loading = true;
            try {
                const response = await axios.get('/api/dashboard/stats');
                this.dashboardStats = response.data;
            } catch (error) {
                this.dashboardStats = {
                    total_images: 0,
                    total_videos: 0,
                    total_size_bytes: 0,
                    duplicates: 0
                };
            } finally {
                this.loading = false;
            }
        },
        
        async loadImages() {
            this.loading = true;
            try {
                const response = await axios.get('/api/images');
                this.images = response.data.images || [];
            } catch (error) {
                console.error('Failed to load images:', error);
                this.images = [];
            } finally {
                this.loading = false;
            }
        },
        
        // Job methods
        async loadJobs() {
            try {
                const response = await axios.get('/api/jobs');
                const jobsList = response.data.jobs || [];
                
                if (Array.isArray(jobsList)) {
                    this.allJobs = jobsList;
                } else if (typeof jobsList === 'object') {
                    this.allJobs = Object.keys(jobsList);
                }
                
                for (const jobId of this.allJobs) {
                    this.loadJobDetails(jobId);
                }
            } catch (error) {
                console.error('Error loading jobs:', error);
            }
        },
        
        async loadJobDetails(jobId) {
            try {
                const response = await axios.get(`/api/jobs/${jobId}`);
                this.jobDetailsCache[jobId] = response.data;
            } catch (error) {
                console.error(`Error loading job ${jobId}:`, error);
            }
        },
        
        async submitAnalyzeJob() {
            try {
                const catalog = this.catalogs.find(c => c.id === this.analyzeForm.catalog_id);
                if (!catalog) {
                    this.addNotification('Please select a catalog', 'error');
                    return;
                }
                
                const response = await axios.post('/api/jobs/analyze', {
                    catalog_path: catalog.catalog_path,
                    source_directories: catalog.source_directories,
                    detect_duplicates: this.analyzeForm.detect_duplicates
                });
                
                this.addNotification('Analysis job submitted successfully', 'success');
                this.showAnalyzeForm = false;
                
                if (response.data.job_id) {
                    this.allJobs.unshift(response.data.job_id);
                    this.loadJobDetails(response.data.job_id);
                }
                
                this.setView('jobs');
            } catch (error) {
                this.addNotification('Failed to submit analysis job: ' + (error.response?.data?.detail || error.message), 'error');
                console.error(error);
            }
        },
        
        async submitOrganizeJob() {
            try {
                const catalog = this.catalogs.find(c => c.id === this.organizeForm.catalog_id);
                if (!catalog) {
                    this.addNotification('Please select a catalog', 'error');
                    return;
                }
                
                const response = await axios.post('/api/jobs/organize', {
                    catalog_path: catalog.catalog_path,
                    output_directory: this.organizeForm.output_directory,
                    dry_run: this.organizeForm.dry_run,
                    operation: this.organizeForm.operation,
                    pattern: this.organizeForm.pattern
                });
                
                this.addNotification('Organization job submitted successfully', 'success');
                this.showOrganizeForm = false;
                
                if (response.data.job_id) {
                    this.allJobs.unshift(response.data.job_id);
                    this.loadJobDetails(response.data.job_id);
                }
                
                this.setView('jobs');
            } catch (error) {
                this.addNotification('Failed to submit organization job: ' + (error.response?.data?.detail || error.message), 'error');
                console.error(error);
            }
        },
        
        async submitThumbnailsJob() {
            try {
                const catalog = this.catalogs.find(c => c.id === this.thumbnailsForm.catalog_id);
                if (!catalog) {
                    this.addNotification('Please select a catalog', 'error');
                    return;
                }
                
                const sizes = this.thumbnailsForm.sizes
                    .split(',')
                    .map(s => parseInt(s.trim()))
                    .filter(n => !isNaN(n));
                
                const response = await axios.post('/api/jobs/thumbnails', {
                    catalog_path: catalog.catalog_path,
                    sizes: sizes,
                    quality: this.thumbnailsForm.quality,
                    skip_existing: this.thumbnailsForm.skip_existing
                });
                
                this.addNotification('Thumbnail generation job submitted successfully', 'success');
                this.showThumbnailsForm = false;
                
                if (response.data.job_id) {
                    this.allJobs.unshift(response.data.job_id);
                    this.loadJobDetails(response.data.job_id);
                }
                
                this.setView('jobs');
            } catch (error) {
                this.addNotification('Failed to submit thumbnails job: ' + (error.response?.data?.detail || error.message), 'error');
                console.error(error);
            }
        },
        
        async cancelJob(jobId) {
            try {
                await axios.delete(`/api/jobs/${jobId}`);
                this.addNotification('Job cancelled', 'info');
                this.loadJobDetails(jobId);
            } catch (error) {
                this.addNotification('Failed to cancel job: ' + (error.response?.data?.detail || error.message), 'error');
                console.error(error);
            }
        },

        async killJob(jobId) {
            if (!confirm('Force kill this job? This may leave files in an incomplete state.')) {
                return;
            }
            
            try {
                await axios.post(`/api/jobs/${jobId}/kill`);
                this.addNotification('Job force-killed. Check catalog integrity.', 'warning');
                this.loadJobDetails(jobId);
            } catch (error) {
                this.addNotification('Failed to kill job: ' + (error.response?.data?.detail || error.message), 'error');
                console.error(error);
            }
        },
        
        isJobStuck(job) {
            // Consider a job stuck if it's been in PROGRESS for > 5 minutes without completion
            if (job.status !== 'PROGRESS') return false;
            
            // Simple heuristic: if no progress change for extended time
            // In a real implementation, you'd track last progress update time
            return false; // Placeholder
        },
        
        startJobsRefresh() {
            if (this.jobsRefreshInterval) return;
            
            this.loadJobs();
            this.jobsRefreshInterval = setInterval(() => {
                for (const job of this.activeJobs) {
                    this.loadJobDetails(job.id);
                }
            }, 2000);
        },
        
        stopJobsRefresh() {
            if (this.jobsRefreshInterval) {
                clearInterval(this.jobsRefreshInterval);
                this.jobsRefreshInterval = null;
            }
        },
        
        getJobStatusClass(status) {
            const statusClasses = {
                'PENDING': 'status-pending',
                'PROGRESS': 'status-progress',
                'SUCCESS': 'status-success',
                'FAILURE': 'status-failure'
            };
            return statusClasses[status] || 'status-unknown';
        },
        
        // Notifications
        addNotification(message, type = 'info') {
            const id = Date.now();
            this.notifications.push({ id, message, type });
            setTimeout(() => {
                this.notifications = this.notifications.filter(n => n.id !== id);
            }, 5000);
        },
        
        removeNotification(id) {
            this.notifications = this.notifications.filter(n => n.id !== id);
        },
        
        // Formatting
        formatNumber(num) {
            return new Intl.NumberFormat().format(num || 0);
        },
        
        formatBytes(bytes) {
            if (!bytes) return '0 B';
            const k = 1024;
            const sizes = ['B', 'KB', 'MB', 'GB', 'TB'];
            const i = Math.floor(Math.log(bytes) / Math.log(k));
            return Math.round(bytes / Math.pow(k, i) * 100) / 100 + ' ' + sizes[i];
        },
        
        formatDate(dateStr) {
            if (!dateStr) return 'Unknown';
            return new Date(dateStr).toLocaleString();
        },

        displayPath(path) {
            // Strip /app prefix to show user-friendly paths
            // /app/catalogs/my-catalog -> ~/catalogs/my-catalog
            // /app/photos/vacation -> ~/photos/vacation
            if (!path) return '';
            if (path.startsWith('/app/')) {
                return '~/' + path.substring(5);
            }
            return path;
        }
    },
    
    mounted() {
        // Load catalogs first
        this.loadCatalogs().then(() => {
            // Then load other data
            this.loadCatalogInfo();
            this.loadDashboardStats();
            this.loadJobs();
            
            // Start monitoring active jobs
            this.startJobsRefresh();
        });
    },
    
    beforeUnmount() {
        this.stopJobsRefresh();
    }
}).mount('#app');
