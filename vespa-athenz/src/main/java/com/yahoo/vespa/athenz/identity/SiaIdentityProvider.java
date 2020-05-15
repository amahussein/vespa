// Copyright 2018 Yahoo Holdings. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.
package com.yahoo.vespa.athenz.identity;

import com.google.inject.Inject;
import com.yahoo.component.AbstractComponent;
import com.yahoo.security.SslContextBuilder;
import com.yahoo.security.X509CertificateWithKey;
import com.yahoo.security.tls.AutoReloadingX509KeyManager;
import com.yahoo.vespa.athenz.api.AthenzIdentity;
import com.yahoo.vespa.athenz.api.AthenzService;
import com.yahoo.vespa.athenz.utils.SiaUtils;

import javax.net.ssl.SSLContext;
import java.nio.file.Path;
import java.nio.file.Paths;

/**
 * A {@link ServiceIdentityProvider} that provides the credentials stored on file system.
 *
 * @author mortent
 * @author bjorncs
 */
public class SiaIdentityProvider extends AbstractComponent implements ServiceIdentityProvider {

    private final AutoReloadingX509KeyManager keyManager;
    private final SSLContext sslContext;
    private final AthenzIdentity service;
    private final Path certificateFile;
    private final Path privateKeyFile;
    private final Path clientTruststoreFile;
    private final Path athenzTruststoreFile;

    @Inject
    public SiaIdentityProvider(SiaProviderConfig config) {
        this(new AthenzService(config.athenzIdentity()),
                Paths.get(config.siaDirectory()),
                Paths.get(config.athenzTruststorePath()),
                Paths.get(config.clientTruststorePath()));
    }

    public SiaIdentityProvider(AthenzIdentity service,
                               Path siaPath,
                               Path clientTruststoreFile,
                               Path athenzTruststoreFile) {
        this(service,
                SiaUtils.getPrivateKeyFile(siaPath, service),
                SiaUtils.getCertificateFile(siaPath, service),
                athenzTruststoreFile,
                clientTruststoreFile);
    }

    public SiaIdentityProvider(AthenzIdentity service,
                               Path privateKeyFile,
                               Path certificateFile,
                               Path athenzTruststoreFile,
                               Path clientTruststoreFile) {
        this.service = service;
        this.keyManager = AutoReloadingX509KeyManager.fromPemFiles(privateKeyFile, certificateFile);
        this.sslContext = createIdentitySslContext(keyManager, clientTruststoreFile);
        this.certificateFile = certificateFile;
        this.privateKeyFile = privateKeyFile;
        this.athenzTruststoreFile = athenzTruststoreFile;
        this.clientTruststoreFile = clientTruststoreFile;
    }

    @Override
    public AthenzIdentity identity() {
        return service;
    }

    @Override
    public SSLContext getIdentitySslContext() {
        return sslContext;
    }

    @Override public X509CertificateWithKey getIdentityCertificateWithKey() { return keyManager.getCurrentCertificateWithKey(); }
    @Override public Path certificatePath() { return certificateFile; }
    @Override public Path privateKeyPath() { return privateKeyFile; }
    @Override public Path athenzTruststorePath() { return athenzTruststoreFile; }
    @Override public Path clientTruststorePath() { return clientTruststoreFile; }

    private static SSLContext createIdentitySslContext(AutoReloadingX509KeyManager keyManager, Path trustStoreFile) {
        return new SslContextBuilder()
                .withTrustStore(trustStoreFile)
                .withKeyManager(keyManager)
                .build();
    }

    @Override
    public void deconstruct() {
        keyManager.close();
    }

}
